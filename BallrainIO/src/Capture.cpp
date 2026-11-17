#include "BallrainIO.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>

// Video capture (0.3.3). Contract (environment, read once at OnLoad; inert without BALLRAIN_CAPTURE):
//   BALLRAIN_CAPTURE=<path>     where the frames go: a named pipe (\\.\pipe\...) opened as a client, or a file
//   BALLRAIN_CAPTURE_FPS=<f>    video frame rate (default 60)
//   BALLRAIN_CAPTURE_TAIL=<n>   agent mode: n idle frames after a finished run's final observation, for the ending
//                               (see CaptureTailStart); the exit waits for them
//
// The game's own picture, read back from its back buffer (CKRenderContext::DumpToMemory) in a post-sprite
// render callback, i.e. after the 3D scene, the 2D interface and every sprite are drawn and before the
// back buffer is shown. Nothing on the screen or desktop is read.
//
// Fixed game-time steps: the run's game time is the sum of the injected frame deltas since its first frame
// (prefix, agent and SR tail / post-finish frames alike). Video frame j is due at j * 1000 / fps ms; a game
// frame whose time passes one or more due points is rendered and written once with the number of video frames
// it stands for, every other frame of the run is not rendered (SkipRenderForNextTick, the training default).
// So the video plays at real game speed whatever the frame deltas are (3.79 ms TAS frames, 7.58 ms agent
// frames). Frames outside the run (menu, before the level load) are left as they are and not written.
//
// Output only: no game object, input, timing or physics state is read or written besides the delta read
// back from the time manager and the back buffer.
//
// Stream: per written frame a 48-byte little-endian header
//   "BRVF", u32 frame index (frames played), u32 repeat (video frames), u32 width, u32 height,
//   u32 bytes per line, u32 bits per pixel, u32 red mask, u32 green mask, u32 blue mask, f64 game time (ms)
// followed by height * bytes-per-line bytes of pixels, rows top to bottom as DumpToMemory returns them.

namespace {
struct CapHeader {
    char magic[4];
    uint32_t frame, repeat, width, height, pitch, bpp, rmask, gmask, bmask;
    double gameMs;
};
static_assert(sizeof(CapHeader) == 48, "capture header is 48 bytes");

std::string Env(const char *name) {
    const char *v = std::getenv(name);
    return v ? std::string(v) : std::string();
}
}  // namespace

void BallrainIO::CaptureInit() {
    m_CapturePath = Env("BALLRAIN_CAPTURE");
    if (m_CapturePath.empty())
        return;
    const std::string fps = Env("BALLRAIN_CAPTURE_FPS");
    if (!fps.empty()) {
        const double f = std::atof(fps.c_str());
        if (f > 0 && f <= 1000)
            m_CaptureFps = f;
    }
    m_CaptureTail = (uint32_t) std::strtoul(Env("BALLRAIN_CAPTURE_TAIL").c_str(), nullptr, 10);
    GetLogger()->Info("capture: %.3g fps to %s, tail %u frames", m_CaptureFps, m_CapturePath.c_str(), m_CaptureTail);
}

// Capture tail (agent mode, BALLRAIN_CAPTURE_TAIL=<n>): after a finished run's final observation (AgentFinish: SR tail
// done, outputs closed, the run's data complete), n more idle frames (keys released, the last frame's delta) are run
// and captured so the video shows the ending; then the game exits as it would have. Nothing the run reports is
// affected: the frame log, trace, events and final observation are all written before the first of these frames.
bool BallrainIO::CaptureTailStart() {
    if (m_CapturePath.empty() || m_CaptureDone || !m_CaptureTail || !m_CaptureFile)
        return false;
    m_CaptureTailActive = true;
    m_CaptureTailLeft = m_CaptureTail;
    GetLogger()->Info("capture: %u tail frames after the final observation", m_CaptureTail);
    return true;
}

bool BallrainIO::CaptureTailTime() {
    if (!m_CaptureTailActive)
        return false;
    m_TimeManager->SetLastDeltaTime(m_LastDelta);
    return true;
}

bool BallrainIO::CaptureTailInput() {
    if (!m_CaptureTailActive)
        return false;
    if (m_InputHook)
        SetKeyboardState(m_InputHook->GetKeyboardState(), 0);
    ++m_FrameIndex;
    if (m_CaptureTailLeft)
        --m_CaptureTailLeft;
    return true;
}

bool BallrainIO::CaptureRegister() {
    if (m_CaptureRc)
        return true;
    CKRenderContext *rc = m_BML->GetRenderContext();
    if (!rc)
        return false;
    rc->AddPostSpriteRenderCallBack(&BallrainIO::CaptureRenderCallback, this, FALSE);
    m_CaptureRc = rc;
    GetLogger()->Info("capture: render callback on a %dx%d render context", rc->GetWidth(), rc->GetHeight());
    return true;
}

void BallrainIO::CaptureClose() {
    if (m_CaptureFile) {
        ::FlushFileBuffers((HANDLE) m_CaptureFile);
        ::CloseHandle((HANDLE) m_CaptureFile);
        m_CaptureFile = nullptr;
        GetLogger()->Info("capture: closed, %u frames rendered, %u written (%lld video frames, %.1f ms game time), %u failed",
                          m_CaptureFrames, m_CaptureWritten, m_CaptureNext, m_CaptureTime, m_CaptureFailed);
    }
    m_CaptureDone = true;
    m_CaptureRepeat = 0;
}

// OnProcess, every frame while the mod is active (before the replay trace line and the agent step).
void BallrainIO::CaptureProcess() {
    m_CaptureRepeat = 0;
    if (m_CapturePath.empty())
        return;
    // The capture tail ends when it is over, or as soon as the capture stopped (pipe gone: the reader died or
    // closed it): close, then the exit AgentFinish deferred. Checked before anything else so the game can never
    // keep running idle frames after the run with nobody reading.
    if (m_CaptureTailActive && (m_CaptureTailLeft == 0 || m_CaptureDone || !m_CaptureFile)) {
        m_CaptureTailActive = false;
        CaptureClose();
        if (m_ExitWhenDone)
            m_BML->ExitGame();
        return;
    }
    if (m_CaptureDone || !CaptureRegister())
        return;
    const bool agentTail = m_Mode == Mode::Agent && (m_Phase == Phase::Tail || m_Phase == Phase::Post);
    const bool inRun = m_Playing || agentTail || m_CaptureTailActive;
    if (!m_CaptureStarted) {
        if (!inRun)
            return;   // menu / level load before the run: rendered as usual, not written
        m_CaptureStarted = true;
        m_CaptureTime = 0.0;
        m_CaptureNext = 0;
        m_CaptureLastIndex = m_FrameIndex;
        if (m_CapturePath.rfind("\\\\.\\pipe\\", 0) == 0) {
            HANDLE h = INVALID_HANDLE_VALUE;
            for (int i = 0; i < 100 && h == INVALID_HANDLE_VALUE; ++i) {
                h = ::CreateFileA(m_CapturePath.c_str(), GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, 0, nullptr);
                if (h == INVALID_HANDLE_VALUE && ::GetLastError() == ERROR_PIPE_BUSY)
                    ::WaitNamedPipeA(m_CapturePath.c_str(), 100);
                else if (h == INVALID_HANDLE_VALUE)
                    break;
            }
            m_CaptureFile = h == INVALID_HANDLE_VALUE ? nullptr : (void *) h;
        } else {
            HANDLE h = ::CreateFileA(m_CapturePath.c_str(), GENERIC_WRITE, 0, nullptr, CREATE_ALWAYS,
                                     FILE_ATTRIBUTE_NORMAL, nullptr);
            m_CaptureFile = h == INVALID_HANDLE_VALUE ? nullptr : (void *) h;
        }
        if (!m_CaptureFile) {
            GetLogger()->Error("capture: cannot open %s (error %lu): capture off", m_CapturePath.c_str(), ::GetLastError());
            m_CaptureDone = true;
            return;
        }
        GetLogger()->Info("capture: started at frame %u", (unsigned) m_FrameIndex);
    }
    if (!inRun) {   // the run is over (replay: stopped; agent: ended)
        CaptureClose();
        return;
    }
    if (m_FrameIndex != m_CaptureLastIndex) {   // a frame was played in this tick
        m_CaptureLastIndex = m_FrameIndex;
        m_CaptureTime += (double) m_TimeManager->GetLastDeltaTime();
        uint32_t due = 0;
        while ((double) m_CaptureNext * 1000.0 / m_CaptureFps <= m_CaptureTime + 1e-9) {
            ++m_CaptureNext;
            ++due;
        }
        m_CaptureRepeat = due;
    }
    if (m_CaptureRepeat == 0)
        m_BML->SkipRenderForNextTick();
}

void BallrainIO::CaptureRenderCallback(CKRenderContext *rc, void *self) {
    auto *mod = static_cast<BallrainIO *>(self);
    if (mod->m_CaptureRepeat && mod->m_CaptureFile)
        mod->CaptureFrame(rc);
    mod->m_CaptureRepeat = 0;
}

void BallrainIO::CaptureFrame(CKRenderContext *rc) {
    ++m_CaptureFrames;
    VxImageDescEx desc;
    const int size = rc->DumpToMemory(nullptr, VXBUFFER_BACKBUFFER, desc);
    bool ok = size > 0 && desc.Width > 0 && desc.Height > 0 && desc.BytesPerLine > 0;
    if (ok) {
        if (m_CaptureBuf.size() < (size_t) size)
            m_CaptureBuf.resize((size_t) size);
        desc.Image = m_CaptureBuf.data();
        ok = rc->DumpToMemory(nullptr, VXBUFFER_BACKBUFFER, desc) > 0;
    }
    if (!ok) {
        if (m_CaptureFailed++ < 5)
            GetLogger()->Error("capture: DumpToMemory failed at frame %u (size %d)", (unsigned) m_FrameIndex, size);
        return;
    }
    CapHeader h = {};
    std::memcpy(h.magic, "BRVF", 4);
    h.frame = (uint32_t) m_FrameIndex;
    h.repeat = m_CaptureRepeat;
    h.width = (uint32_t) desc.Width;
    h.height = (uint32_t) desc.Height;
    h.pitch = (uint32_t) desc.BytesPerLine;
    h.bpp = (uint32_t) desc.BitsPerPixel;
    h.rmask = desc.RedMask;
    h.gmask = desc.GreenMask;
    h.bmask = desc.BlueMask;
    h.gameMs = m_CaptureTime;
    const size_t bytes = (size_t) desc.Height * (size_t) desc.BytesPerLine;
    auto write = [this](const void *p, size_t n) {
        const auto *c = static_cast<const char *>(p);
        while (n > 0) {
            DWORD done = 0;
            const DWORD chunk = (DWORD) std::min<size_t>(n, 1u << 20);
            if (!::WriteFile((HANDLE) m_CaptureFile, c, chunk, &done, nullptr) || done == 0)
                return false;
            c += done;
            n -= done;
        }
        return true;
    };
    if (bytes > m_CaptureBuf.size() || !write(&h, sizeof(h)) || !write(m_CaptureBuf.data(), bytes)) {
        GetLogger()->Error("capture: write failed at frame %u (error %lu): capture off", (unsigned) m_FrameIndex,
                           ::GetLastError());
        CaptureClose();
        return;
    }
    ++m_CaptureWritten;
}
