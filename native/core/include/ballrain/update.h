// Update check against a signed release manifest (docs/native-client.md section 8): release.json next to the release
// zips (GitHub Releases: .../releases/latest/download/release.json) and release.json.sig, the hex ed25519 signature
// of the exact manifest bytes by the release key (tools: nc_release keygen / sign). The client only reports a newer
// version (banner + the download link); it does not replace itself yet.
//
// release.json: {"product": "ballrain-pool", "version": "0.3.1", "notes": "...",
//                "assets": {"windows-x64": {"url": "...", "sha256": "...", "size": n}, ...}}
#pragma once

#include <optional>
#include <string>

namespace ballrain {

struct ReleaseInfo {
    std::string version, notes, url, sha256;
    long long size = 0;
};

// -1 / 0 / 1 for dotted numeric versions ("0.3.10" > "0.3.9"); a "-suffix" sorts before the plain version
int compare_versions(const std::string &a, const std::string &b);
bool verify_release_signature(const std::string &manifest, const std::string &sig_hex, const std::string &pubkey_hex);
// The newer release for this platform, or nullopt (none newer, or error set: unreachable, bad signature, bad format)
std::optional<ReleaseInfo> check_for_update(const std::string &manifest_url, const std::string &pubkey_hex,
                                            const std::string &current_version, const std::string &platform_key,
                                            std::string *error);
std::string release_platform_key();  // "windows-x64", "macos-arm64", "linux-x64", ...

}  // namespace ballrain
