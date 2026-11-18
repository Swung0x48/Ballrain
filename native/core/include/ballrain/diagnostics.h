// Diagnostics export: what a tester sends when something goes wrong (logs, configuration, versions, the game-file
// hashes), with private data replaced before anything is shown or saved: the token, the home directory and the login
// name in paths, the forum account, IPv4 addresses. The TUI shows the file list and info.json before it saves.
#pragma once

#include <string>
#include <utility>
#include <vector>

namespace ballrain {

using DiagFiles = std::vector<std::pair<std::string, std::string>>;  // name in the zip -> content

// text with every replace.first (case-insensitively for ASCII letters; "\" and "/" alike) replaced by replace.second,
// longest first, and IPv4 addresses (a.b.c.d) replaced by <ip>
std::string scrub_text(const std::string &text, std::vector<std::pair<std::string, std::string>> replace);
void write_zip(const DiagFiles &files, const std::string &path);  // throws Error

}  // namespace ballrain
