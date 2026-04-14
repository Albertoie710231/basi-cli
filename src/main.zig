const std = @import("std");
const llama = @import("llama.zig");
const c = llama.c;

const MAX_TOKENS = 8192;
const CONTEXT_SIZE = 4096;
const MAX_FILE_TOKENS = 2000;

// System prompt with tool instructions
const SYSTEM_PROMPT =
    \\You are BASI, a helpful AI assistant with access to file and web tools.
    \\
    \\When you need to read files, search the web, or fetch URLs, use these tools by wrapping commands in <tool> tags:
    \\
    \\FILE TOOLS:
    \\- read <file> : Read entire file (only for small files <2000 tokens)
    \\- head -n <N> <file> : Read first N lines
    \\- tail -n <N> <file> : Read last N lines
    \\- grep <pattern> <file> : Search for pattern (use quotes for multi-word)
    \\- grep -n <pattern> <file> : Search with line numbers
    \\- grep -C <N> <pattern> <file> : Search with N lines of context
    \\- wc <file> : Count lines, words, characters
    \\
    \\WEB TOOLS:
    \\- search <query> : Search DuckDuckGo for information (use for current events, facts, documentation)
    \\- webfetch <url> : Fetch and extract text content from a URL
    \\
    \\Examples:
    \\<tool>head -n 50 /path/to/file.txt</tool>
    \\<tool>grep -n "function main" src/main.zig</tool>
    \\<tool>search zig programming language documentation</tool>
    \\<tool>webfetch https://example.com/docs</tool>
    \\
    \\Tool results will appear in <tool_result> tags. You can use multiple tools to explore large files or gather web information.
    \\For large files, first use 'wc' to check size, then 'head' or 'grep' to read relevant parts.
    \\For web searches, use 'search' first to find relevant URLs, then 'webfetch' to get detailed content.
    \\
    \\Always be helpful, concise, and accurate.
;

// Thinking animation frames
const spinner_frames = [_][]const u8{ "⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏" };

// Log callback to suppress llama.cpp output (only show errors)
fn logCallback(level: c.ggml_log_level, text: [*c]const u8, _: ?*anyopaque) callconv(.c) void {
    if (level >= c.GGML_LOG_LEVEL_ERROR) {
        var buf: [4096]u8 = undefined;
        var stderr_writer = std.fs.File.stderr().writer(&buf);
        stderr_writer.interface.writeAll(std.mem.span(text)) catch {};
        stderr_writer.interface.flush() catch {};
    }
}

// Terminal raw mode for line editing
const Termios = extern struct {
    c_iflag: u32,
    c_oflag: u32,
    c_cflag: u32,
    c_lflag: u32,
    c_line: u8,
    c_cc: [32]u8,
    c_ispeed: u32,
    c_ospeed: u32,
};

extern "c" fn tcgetattr(fd: c_int, termios: *Termios) c_int;
extern "c" fn tcsetattr(fd: c_int, action: c_int, termios: *const Termios) c_int;

const STDIN_FILENO: c_int = 0;
const TCSAFLUSH: c_int = 2;
const ICANON: u32 = 0o0000002;
const ECHO: u32 = 0o0000010;
const ISIG: u32 = 0o0000001;

// Signal handling for Ctrl+C during generation
var generation_interrupted: bool = false;

const sigaction_t = extern struct {
    handler: ?*const fn (c_int) callconv(.c) void,
    mask: [128 / @sizeOf(c_ulong)]c_ulong,
    flags: c_int,
    restorer: ?*const fn () callconv(.c) void,
};

extern "c" fn sigaction(sig: c_int, act: ?*const sigaction_t, oact: ?*sigaction_t) c_int;

const SIGINT: c_int = 2;
const SA_RESTART: c_int = 0x10000000;

fn sigintHandler(_: c_int) callconv(.c) void {
    generation_interrupted = true;
}

fn setupSigintHandler() void {
    var sa: sigaction_t = .{
        .handler = sigintHandler,
        .mask = .{0} ** (128 / @sizeOf(c_ulong)),
        .flags = SA_RESTART,
        .restorer = null,
    };
    _ = sigaction(SIGINT, &sa, null);
}

fn resetSigintHandler() void {
    var sa: sigaction_t = .{
        .handler = null, // SIG_DFL
        .mask = .{0} ** (128 / @sizeOf(c_ulong)),
        .flags = 0,
        .restorer = null,
    };
    _ = sigaction(SIGINT, &sa, null);
}

// Execute a tool command and return result
fn executeTool(allocator: std.mem.Allocator, command: []const u8) ![]u8 {
    // Parse the command
    const trimmed = std.mem.trim(u8, command, " \t\n\r");
    if (trimmed.len == 0) {
        return try allocator.dupe(u8, "Error: Empty command");
    }

    // Tokenize command (simple space-based, respecting quotes)
    var args_list: std.ArrayList([]const u8) = .empty;
    defer args_list.deinit(allocator);

    var i: usize = 0;
    while (i < trimmed.len) {
        // Skip whitespace
        while (i < trimmed.len and (trimmed[i] == ' ' or trimmed[i] == '\t')) : (i += 1) {}
        if (i >= trimmed.len) break;

        var start = i;
        var end = i;

        if (trimmed[i] == '"' or trimmed[i] == '\'') {
            // Quoted argument
            const quote = trimmed[i];
            i += 1;
            start = i;
            while (i < trimmed.len and trimmed[i] != quote) : (i += 1) {}
            end = i;
            if (i < trimmed.len) i += 1; // Skip closing quote
        } else {
            // Unquoted argument
            while (i < trimmed.len and trimmed[i] != ' ' and trimmed[i] != '\t') : (i += 1) {}
            end = i;
        }

        if (end > start) {
            try args_list.append(allocator, trimmed[start..end]);
        }
    }

    if (args_list.items.len == 0) {
        return try allocator.dupe(u8, "Error: No command specified");
    }

    const cmd = args_list.items[0];

    // Validate command (whitelist approach)
    const allowed_commands = [_][]const u8{ "read", "head", "tail", "grep", "wc", "cat", "search", "webfetch" };
    var is_allowed = false;
    for (allowed_commands) |allowed| {
        if (std.mem.eql(u8, cmd, allowed)) {
            is_allowed = true;
            break;
        }
    }

    if (!is_allowed) {
        return try std.fmt.allocPrint(allocator, "Error: Command '{s}' not allowed. Use: read, head, tail, grep, wc, search, webfetch", .{cmd});
    }

    // Handle 'read' as 'cat' with size check
    if (std.mem.eql(u8, cmd, "read")) {
        if (args_list.items.len < 2) {
            return try allocator.dupe(u8, "Error: read requires a file path");
        }
        const filepath = args_list.items[1];

        // Check file size first
        const file = std.fs.cwd().openFile(filepath, .{}) catch |err| {
            return try std.fmt.allocPrint(allocator, "Error: Cannot open file '{s}': {}", .{ filepath, err });
        };
        defer file.close();

        const stat = file.stat() catch |err| {
            return try std.fmt.allocPrint(allocator, "Error: Cannot stat file: {}", .{err});
        };

        // Rough estimate: ~4 chars per token
        const estimated_tokens = stat.size / 4;
        if (estimated_tokens > MAX_FILE_TOKENS) {
            return try std.fmt.allocPrint(
                allocator,
                "Error: File too large (~{d} tokens, max {d}). Use 'head', 'tail', or 'grep' to read in chunks.\nFile has {d} bytes, {d} lines (use 'wc {s}' for exact count)",
                .{ estimated_tokens, MAX_FILE_TOKENS, stat.size, countLines(file), filepath },
            );
        }

        // Read the file
        const content = file.readToEndAlloc(allocator, 1024 * 1024) catch |err| {
            return try std.fmt.allocPrint(allocator, "Error: Cannot read file: {}", .{err});
        };
        return content;
    }

    // Handle 'search' - DuckDuckGo web search
    if (std.mem.eql(u8, cmd, "search")) {
        if (args_list.items.len < 2) {
            return try allocator.dupe(u8, "Error: search requires a query");
        }

        // Build query from all remaining arguments
        var query: std.ArrayList(u8) = .empty;
        defer query.deinit(allocator);
        for (args_list.items[1..], 0..) |arg, idx| {
            if (idx > 0) try query.append(allocator, ' ');
            try query.appendSlice(allocator, arg);
        }

        return try executeWebSearch(allocator, query.items);
    }

    // Handle 'webfetch' - fetch URL content
    if (std.mem.eql(u8, cmd, "webfetch")) {
        if (args_list.items.len < 2) {
            return try allocator.dupe(u8, "Error: webfetch requires a URL");
        }
        return try executeWebFetch(allocator, args_list.items[1]);
    }

    // Build actual command for shell execution
    var shell_cmd: std.ArrayList(u8) = .empty;
    defer shell_cmd.deinit(allocator);

    // Map 'read' to 'cat' if it somehow got here
    const actual_cmd = if (std.mem.eql(u8, cmd, "read")) "cat" else cmd;
    try shell_cmd.appendSlice(allocator, actual_cmd);

    for (args_list.items[1..]) |arg| {
        try shell_cmd.append(allocator, ' ');
        // Quote arguments that need it
        if (std.mem.indexOfAny(u8, arg, " \t\"'$`\\")) |_| {
            try shell_cmd.append(allocator, '\'');
            for (arg) |ch| {
                if (ch == '\'') {
                    try shell_cmd.appendSlice(allocator, "'\"'\"'");
                } else {
                    try shell_cmd.append(allocator, ch);
                }
            }
            try shell_cmd.append(allocator, '\'');
        } else {
            try shell_cmd.appendSlice(allocator, arg);
        }
    }

    // Execute via shell
    const shell_cmd_z = try allocator.dupeZ(u8, shell_cmd.items);
    defer allocator.free(shell_cmd_z);

    var child = std.process.Child.init(&[_][]const u8{ "/bin/sh", "-c", shell_cmd_z }, allocator);
    child.stdout_behavior = .Pipe;
    child.stderr_behavior = .Pipe;

    try child.spawn();

    const stdout_result = child.stdout.?.readToEndAlloc(allocator, 512 * 1024) catch |err| {
        return try std.fmt.allocPrint(allocator, "Error reading stdout: {}", .{err});
    };
    const stderr_result = child.stderr.?.readToEndAlloc(allocator, 64 * 1024) catch "";
    defer allocator.free(stderr_result);

    const term = child.wait() catch |err| {
        allocator.free(stdout_result);
        return try std.fmt.allocPrint(allocator, "Error waiting for command: {}", .{err});
    };

    if (term.Exited != 0 and stderr_result.len > 0) {
        allocator.free(stdout_result);
        return try std.fmt.allocPrint(allocator, "Error: {s}", .{stderr_result});
    }

    if (stdout_result.len == 0 and stderr_result.len > 0) {
        allocator.free(stdout_result);
        return try allocator.dupe(u8, stderr_result);
    }

    return stdout_result;
}

fn countLines(file: std.fs.File) usize {
    file.seekTo(0) catch return 0;
    var count: usize = 0;
    var buf: [4096]u8 = undefined;
    while (true) {
        const n = file.read(&buf) catch break;
        if (n == 0) break;
        for (buf[0..n]) |ch| {
            if (ch == '\n') count += 1;
        }
    }
    file.seekTo(0) catch {};
    return count;
}

// URL-encode a string for use in query parameters
fn urlEncode(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    var result: std.ArrayList(u8) = .empty;
    errdefer result.deinit(allocator);

    for (input) |ch| {
        if ((ch >= 'a' and ch <= 'z') or (ch >= 'A' and ch <= 'Z') or (ch >= '0' and ch <= '9') or ch == '-' or ch == '_' or ch == '.' or ch == '~') {
            try result.append(allocator, ch);
        } else if (ch == ' ') {
            try result.append(allocator, '+');
        } else {
            try result.append(allocator, '%');
            const hex = "0123456789ABCDEF";
            try result.append(allocator, hex[ch >> 4]);
            try result.append(allocator, hex[ch & 0x0F]);
        }
    }

    return try result.toOwnedSlice(allocator);
}

// Execute DuckDuckGo web search using mini_browser
fn executeWebSearch(allocator: std.mem.Allocator, query: []const u8) ![]u8 {
    const encoded_query = try urlEncode(allocator, query);
    defer allocator.free(encoded_query);

    // Build DuckDuckGo HTML lite URL
    const url = try std.fmt.allocPrint(allocator, "https://lite.duckduckgo.com/lite/?q={s}", .{encoded_query});
    defer allocator.free(url);

    // Use mini_browser to fetch the page (bypasses CAPTCHAs)
    // Filter out ads (contain ad_domain or ad_provider) and help pages
    const browser_cmd = try std.fmt.allocPrint(
        allocator,
        "/home/alberto/Documentos/MINI_BROWSER/build/mini_browser --headless --timeout 30000 '{s}' 2>/dev/null | grep -oP '(?<=class=\"result-link\" href=\")[^\"]+' | grep -v 'ad_domain' | grep -v 'ad_provider' | grep -v 'duckduckgo.com/duckduckgo-help-pages' | head -15",
        .{url},
    );
    defer allocator.free(browser_cmd);

    const browser_cmd_z = try allocator.dupeZ(u8, browser_cmd);
    defer allocator.free(browser_cmd_z);

    var child = std.process.Child.init(&[_][]const u8{ "/bin/sh", "-c", browser_cmd_z }, allocator);
    child.stdout_behavior = .Pipe;
    child.stderr_behavior = .Pipe;

    try child.spawn();

    const stdout_result = child.stdout.?.readToEndAlloc(allocator, 256 * 1024) catch |err| {
        return try std.fmt.allocPrint(allocator, "Error fetching search results: {}", .{err});
    };
    const stderr_result = child.stderr.?.readToEndAlloc(allocator, 16 * 1024) catch "";
    defer allocator.free(stderr_result);

    const term = child.wait() catch |err| {
        allocator.free(stdout_result);
        return try std.fmt.allocPrint(allocator, "Error waiting for browser: {}", .{err});
    };

    if (term.Exited != 0 and stdout_result.len == 0) {
        allocator.free(stdout_result);
        return try std.fmt.allocPrint(allocator, "Search failed: {s}", .{stderr_result});
    }

    if (stdout_result.len == 0) {
        return try allocator.dupe(u8, "No search results found.");
    }

    // Parse DDG redirect URLs to extract actual URLs
    var results: std.ArrayList(u8) = .empty;
    errdefer results.deinit(allocator);

    var lines = std.mem.splitScalar(u8, stdout_result, '\n');
    var count: usize = 0;
    while (lines.next()) |line| {
        if (line.len == 0) continue;
        if (count >= 10) break;

        // Extract actual URL from DDG redirect (uddg= parameter)
        if (std.mem.indexOf(u8, line, "uddg=")) |uddg_start| {
            const url_start = uddg_start + 5;
            const url_end = std.mem.indexOfPos(u8, line, url_start, "&") orelse line.len;
            const encoded_url = line[url_start..url_end];

            // URL decode the result
            const decoded = try urlDecode(allocator, encoded_url);
            defer allocator.free(decoded);

            try results.appendSlice(allocator, decoded);
            try results.append(allocator, '\n');
            count += 1;
        }
    }

    allocator.free(stdout_result);

    if (results.items.len == 0) {
        results.deinit(allocator);
        return try allocator.dupe(u8, "No search results found.");
    }

    return try results.toOwnedSlice(allocator);
}

// URL decode a string
fn urlDecode(allocator: std.mem.Allocator, input: []const u8) ![]u8 {
    var result: std.ArrayList(u8) = .empty;
    errdefer result.deinit(allocator);

    var i: usize = 0;
    while (i < input.len) {
        if (input[i] == '%' and i + 2 < input.len) {
            const hex = input[i + 1 .. i + 3];
            const byte = std.fmt.parseInt(u8, hex, 16) catch {
                try result.append(allocator, input[i]);
                i += 1;
                continue;
            };
            try result.append(allocator, byte);
            i += 3;
        } else if (input[i] == '+') {
            try result.append(allocator, ' ');
            i += 1;
        } else {
            try result.append(allocator, input[i]);
            i += 1;
        }
    }

    return try result.toOwnedSlice(allocator);
}

// Execute URL fetch using mini_browser
fn executeWebFetch(allocator: std.mem.Allocator, url: []const u8) ![]u8 {
    // Validate URL starts with http:// or https://
    if (!std.mem.startsWith(u8, url, "http://") and !std.mem.startsWith(u8, url, "https://")) {
        return try allocator.dupe(u8, "Error: URL must start with http:// or https://");
    }

    // Use mini_browser to fetch the page and extract text
    const browser_cmd = try std.fmt.allocPrint(
        allocator,
        "/home/alberto/Documentos/MINI_BROWSER/build/mini_browser --headless --timeout 30000 '{s}' 2>/dev/null | sed 's/<script[^>]*>.*<\\/script>//gi' | sed 's/<style[^>]*>.*<\\/style>//gi' | sed 's/<[^>]*>//g' | sed 's/&nbsp;/ /g' | sed 's/&amp;/\\&/g' | sed 's/&lt;/</g' | sed 's/&gt;/>/g' | sed 's/&quot;/\"/g' | grep -v '^[[:space:]]*$' | head -200",
        .{url},
    );
    defer allocator.free(browser_cmd);

    const browser_cmd_z = try allocator.dupeZ(u8, browser_cmd);
    defer allocator.free(browser_cmd_z);

    var child = std.process.Child.init(&[_][]const u8{ "/bin/sh", "-c", browser_cmd_z }, allocator);
    child.stdout_behavior = .Pipe;
    child.stderr_behavior = .Pipe;

    try child.spawn();

    const stdout_result = child.stdout.?.readToEndAlloc(allocator, 512 * 1024) catch |err| {
        return try std.fmt.allocPrint(allocator, "Error fetching URL: {}", .{err});
    };
    const stderr_result = child.stderr.?.readToEndAlloc(allocator, 16 * 1024) catch "";
    defer allocator.free(stderr_result);

    const term = child.wait() catch |err| {
        allocator.free(stdout_result);
        return try std.fmt.allocPrint(allocator, "Error waiting for browser: {}", .{err});
    };

    if (term.Exited != 0 and stdout_result.len == 0) {
        allocator.free(stdout_result);
        return try std.fmt.allocPrint(allocator, "Fetch failed: {s}", .{stderr_result});
    }

    if (stdout_result.len == 0) {
        return try allocator.dupe(u8, "No content retrieved from URL.");
    }

    return stdout_result;
}

// Extract tool command from response (returns null if no tool call found)
fn extractToolCall(text: []const u8) ?[]const u8 {
    const start_tag = "<tool>";
    const end_tag = "</tool>";

    const start_idx = std.mem.indexOf(u8, text, start_tag) orelse return null;
    const content_start = start_idx + start_tag.len;
    const end_idx = std.mem.indexOfPos(u8, text, content_start, end_tag) orelse return null;

    return text[content_start..end_idx];
}

fn enableRawMode(orig: *Termios) bool {
    if (tcgetattr(STDIN_FILENO, orig) < 0) return false;
    var raw = orig.*;
    raw.c_lflag &= ~(ICANON | ECHO | ISIG);
    raw.c_cc[6] = 1;  // VMIN
    raw.c_cc[5] = 0;  // VTIME
    return tcsetattr(STDIN_FILENO, TCSAFLUSH, &raw) >= 0;
}

fn disableRawMode(orig: *const Termios) void {
    _ = tcsetattr(STDIN_FILENO, TCSAFLUSH, orig);
}

// Command history
var history: std.ArrayList([]u8) = .empty;
var history_allocator: ?std.mem.Allocator = null;

fn addToHistory(allocator: std.mem.Allocator, line_text: []const u8) !void {
    if (history_allocator == null) {
        history_allocator = allocator;
    }
    // Don't add empty lines or duplicates of last entry
    if (line_text.len == 0) return;
    if (history.items.len > 0) {
        if (std.mem.eql(u8, history.items[history.items.len - 1], line_text)) return;
    }
    const copy = try allocator.dupe(u8, line_text);
    try history.append(allocator, copy);
    // Limit history size
    if (history.items.len > 100) {
        allocator.free(history.items[0]);
        _ = history.orderedRemove(0);
    }
}

// Line editor with cursor movement and history
fn readLine(allocator: std.mem.Allocator, writer: anytype, prompt: []const u8) !?[]u8 {
    var orig_termios: Termios = undefined;
    const raw_enabled = enableRawMode(&orig_termios);
    defer if (raw_enabled) disableRawMode(&orig_termios);

    var line: std.ArrayList(u8) = .empty;
    defer line.deinit(allocator);
    var cursor: usize = 0;
    var history_index: usize = history.items.len; // Start past end (current input)
    var saved_line: std.ArrayList(u8) = .empty; // Save current input when browsing history
    defer saved_line.deinit(allocator);

    // Print prompt
    try writer.writeAll(prompt);
    try writer.flush();

    const stdin_file = std.fs.File.stdin();

    while (true) {
        var byte: [1]u8 = undefined;
        const n = stdin_file.read(&byte) catch 0;
        if (n == 0) {
            // EOF
            if (line.items.len == 0) return null;
            break;
        }

        const ch = byte[0];

        if (ch == '\n' or ch == '\r') {
            try writer.writeAll("\n");
            try writer.flush();
            break;
        } else if (ch == 27) {
            // Escape sequence
            var seq: [2]u8 = undefined;
            if (stdin_file.read(&seq) catch 0 < 2) continue;
            if (seq[0] == '[') {
                switch (seq[1]) {
                    'A' => { // Up arrow - history previous
                        if (history.items.len > 0 and history_index > 0) {
                            // Save current line if at end
                            if (history_index == history.items.len) {
                                saved_line.clearRetainingCapacity();
                                try saved_line.appendSlice(allocator, line.items);
                            }
                            history_index -= 1;
                            // Clear current line on screen
                            if (cursor > 0) {
                                try writer.print("\x1b[{d}D", .{cursor});
                            }
                            try writer.writeAll("\x1b[K"); // Clear to end
                            // Replace with history entry
                            line.clearRetainingCapacity();
                            try line.appendSlice(allocator, history.items[history_index]);
                            cursor = line.items.len;
                            try writer.writeAll(line.items);
                            try writer.flush();
                        }
                    },
                    'B' => { // Down arrow - history next
                        if (history_index < history.items.len) {
                            history_index += 1;
                            // Clear current line on screen
                            if (cursor > 0) {
                                try writer.print("\x1b[{d}D", .{cursor});
                            }
                            try writer.writeAll("\x1b[K"); // Clear to end
                            // Replace with history entry or saved line
                            line.clearRetainingCapacity();
                            if (history_index < history.items.len) {
                                try line.appendSlice(allocator, history.items[history_index]);
                            } else {
                                try line.appendSlice(allocator, saved_line.items);
                            }
                            cursor = line.items.len;
                            try writer.writeAll(line.items);
                            try writer.flush();
                        }
                    },
                    'C' => { // Right arrow
                        if (cursor < line.items.len) {
                            cursor += 1;
                            try writer.writeAll("\x1b[C");
                            try writer.flush();
                        }
                    },
                    'D' => { // Left arrow
                        if (cursor > 0) {
                            cursor -= 1;
                            try writer.writeAll("\x1b[D");
                            try writer.flush();
                        }
                    },
                    'H' => { // Home
                        if (cursor > 0) {
                            try writer.print("\x1b[{d}D", .{cursor});
                            try writer.flush();
                            cursor = 0;
                        }
                    },
                    'F' => { // End
                        if (cursor < line.items.len) {
                            try writer.print("\x1b[{d}C", .{line.items.len - cursor});
                            try writer.flush();
                            cursor = line.items.len;
                        }
                    },
                    '3' => { // Delete key (sends [3~)
                        var tilde: [1]u8 = undefined;
                        _ = stdin_file.read(&tilde) catch 0;
                        if (cursor < line.items.len) {
                            _ = line.orderedRemove(cursor);
                            // Redraw from cursor
                            try writer.writeAll("\x1b[s"); // Save cursor
                            try writer.writeAll(line.items[cursor..]);
                            try writer.writeAll(" \x1b[u"); // Clear extra char, restore cursor
                            try writer.flush();
                        }
                    },
                    else => {},
                }
            }
        } else if (ch == 127 or ch == 8) {
            // Backspace
            if (cursor > 0) {
                cursor -= 1;
                _ = line.orderedRemove(cursor);
                // Move back, redraw, clear trailing char
                try writer.writeAll("\x1b[D\x1b[s");
                try writer.writeAll(line.items[cursor..]);
                try writer.writeAll(" \x1b[u");
                try writer.flush();
            }
        } else if (ch == 3) {
            // Ctrl+C - cancel current line, return empty
            try writer.writeAll("^C\n");
            try writer.flush();
            line.clearRetainingCapacity();
            return try line.toOwnedSlice(allocator);
        } else if (ch == 4) {
            // Ctrl+D
            if (line.items.len == 0) return null;
        } else if (ch >= 32) {
            // Printable character
            try line.insert(allocator, cursor, ch);
            cursor += 1;
            if (cursor == line.items.len) {
                // Append at end - simple case
                try writer.writeByte(ch);
                try writer.flush();
            } else {
                // Insert in middle - redraw rest of line
                try writer.writeAll("\x1b[s"); // Save cursor
                try writer.writeByte(ch);
                try writer.writeAll(line.items[cursor..]);
                try writer.writeAll("\x1b[u\x1b[C"); // Restore cursor, move right
                try writer.flush();
            }
        }
    }

    // Add to history before returning
    try addToHistory(allocator, line.items);
    return try line.toOwnedSlice(allocator);
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Zig 0.15+ buffered I/O
    var stdout_buf: [4096]u8 = undefined;
    var stdout_writer = std.fs.File.stdout().writer(&stdout_buf);
    const stdout = &stdout_writer.interface;

    // Get model path from args or environment
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    var model_path: ?[:0]const u8 = null;
    var n_gpu_layers: i32 = 99;

    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        if (std.mem.eql(u8, args[i], "-m") and i + 1 < args.len) {
            i += 1;
            model_path = args[i];
        } else if (std.mem.eql(u8, args[i], "-ngl") and i + 1 < args.len) {
            i += 1;
            n_gpu_layers = std.fmt.parseInt(i32, args[i], 10) catch 99;
        } else if (std.mem.eql(u8, args[i], "-h") or std.mem.eql(u8, args[i], "--help")) {
            try stdout.print("BASI-CLI - AI Chat Interface\n\nUsage: basi-cli -m <model.gguf> [-ngl <gpu_layers>]\n\n  -m     Path to GGUF model file\n  -ngl   Number of GPU layers (default: 99)\n  -h     Show this help\n\nEnvironment:\n  BASI_MODEL  Default model path if -m not specified\n\n", .{});
            try stdout.flush();
            return;
        }
    }

    if (model_path == null) {
        if (std.posix.getenv("BASI_MODEL")) |env_path| {
            model_path = env_path;
        } else {
            try stdout.print("Error: No model specified. Use -m <model.gguf> or set BASI_MODEL\n", .{});
            try stdout.flush();
            return;
        }
    }

    try stdout.print("BASI-CLI - Loading model...\n", .{});
    try stdout.flush();

    // Suppress llama.cpp log output (only show errors)
    c.llama_log_set(logCallback, null);

    // Load backends
    c.ggml_backend_load_all();

    // Load model
    var model_params = llama.defaultModelParams();
    model_params.n_gpu_layers = n_gpu_layers;

    const model = c.llama_model_load_from_file(model_path.?, model_params) orelse {
        try stdout.print("Error: Failed to load model from {s}\n", .{model_path.?});
        try stdout.flush();
        return;
    };
    defer c.llama_model_free(model);

    const vocab = c.llama_model_get_vocab(model) orelse {
        try stdout.print("Error: Failed to get vocabulary from model\n", .{});
        try stdout.flush();
        return;
    };

    // Create context
    var ctx_params = llama.defaultContextParams();
    ctx_params.n_ctx = CONTEXT_SIZE;
    ctx_params.n_batch = CONTEXT_SIZE;

    const ctx = c.llama_init_from_model(model, ctx_params) orelse {
        try stdout.print("Error: Failed to create context\n", .{});
        try stdout.flush();
        return;
    };
    defer c.llama_free(ctx);

    // Create sampler chain
    const smpl = c.llama_sampler_chain_init(llama.defaultSamplerChainParams());
    defer c.llama_sampler_free(smpl);
    c.llama_sampler_chain_add(smpl, c.llama_sampler_init_min_p(0.05, 1));
    c.llama_sampler_chain_add(smpl, c.llama_sampler_init_temp(0.8));
    c.llama_sampler_chain_add(smpl, c.llama_sampler_init_dist(c.LLAMA_DEFAULT_SEED));

    try stdout.print("Model loaded. Type your message (empty line to quit).\n\n", .{});
    try stdout.flush();

    // Chat state (Zig 0.15+ unmanaged ArrayList)
    var messages: std.ArrayList(c.llama_chat_message) = .empty;
    defer {
        for (messages.items) |msg| {
            if (msg.content != null) {
                allocator.free(std.mem.span(msg.content));
            }
        }
        messages.deinit(allocator);
    }

    // Add system prompt
    const system_content = try allocator.dupeZ(u8, SYSTEM_PROMPT);
    try messages.append(allocator, .{
        .role = "system",
        .content = system_content.ptr,
    });

    var formatted_buf: [MAX_TOKENS * 4]u8 = undefined;
    var prev_len: usize = 0;

    // REPL loop
    while (true) {
        // Read user input with line editing
        const user_input = try readLine(allocator, stdout, "\x1b[32m> \x1b[0m") orelse break;
        defer allocator.free(user_input);

        // Empty input (Ctrl+C or just Enter) - show new prompt
        if (user_input.len == 0) continue;

        // Create null-terminated copy for C API
        const user_content = try allocator.dupeZ(u8, user_input);

        // Add user message
        try messages.append(allocator, .{
            .role = "user",
            .content = user_content.ptr,
        });

        // Apply chat template
        const tmpl = c.llama_model_chat_template(model, null);
        const new_len = c.llama_chat_apply_template(
            tmpl,
            messages.items.ptr,
            messages.items.len,
            true,
            &formatted_buf,
            formatted_buf.len,
        );

        if (new_len < 0) {
            try stdout.print("Error: Failed to apply chat template\n", .{});
            try stdout.flush();
            continue;
        }

        // Extract just the new prompt portion
        var prompt = formatted_buf[prev_len..@intCast(new_len)];

        // Tool execution loop - keep generating until no more tool calls
        var tool_iterations: usize = 0;
        const max_tool_iterations = 10; // Prevent infinite loops

        while (tool_iterations < max_tool_iterations) : (tool_iterations += 1) {
            // Generate response (with Ctrl+C handling)
            generation_interrupted = false;
            setupSigintHandler();
            const result = try generate(allocator, ctx, vocab, smpl, prompt, stdout);
            resetSigintHandler();

            // Display performance metrics
            const prompt_tps = if (result.prompt_time_ns > 0)
                @as(f64, @floatFromInt(result.prompt_tokens)) / (@as(f64, @floatFromInt(result.prompt_time_ns)) / 1_000_000_000.0)
            else
                0.0;
            const gen_tps = if (result.gen_time_ns > 0)
                @as(f64, @floatFromInt(result.gen_tokens)) / (@as(f64, @floatFromInt(result.gen_time_ns)) / 1_000_000_000.0)
            else
                0.0;
            try stdout.print("\x1b[90m[ Prompt: {d:.1} t/s | Generation: {d:.1} t/s ]\x1b[0m\n", .{ prompt_tps, gen_tps });
            try stdout.flush();

            // Check for tool call in response
            if (extractToolCall(result.text)) |tool_cmd| {
                // Execute tool
                try stdout.print("\x1b[90m[Executing: {s}]\x1b[0m\n", .{tool_cmd});
                try stdout.flush();

                const tool_result = try executeTool(allocator, tool_cmd);
                defer allocator.free(tool_result);

                // Add assistant response to history
                const response_content = try allocator.dupeZ(u8, result.text);
                allocator.free(result.text);
                try messages.append(allocator, .{
                    .role = "assistant",
                    .content = response_content.ptr,
                });

                // Add tool result as user message (simulating tool response)
                const tool_response_str = try std.fmt.allocPrint(
                    allocator,
                    "<tool_result>\n{s}\n</tool_result>",
                    .{tool_result},
                );
                const tool_response = try allocator.dupeZ(u8, tool_response_str);
                allocator.free(tool_response_str);
                try messages.append(allocator, .{
                    .role = "user",
                    .content = tool_response.ptr,
                });

                // Update template for next iteration
                const next_len = c.llama_chat_apply_template(
                    tmpl,
                    messages.items.ptr,
                    messages.items.len,
                    true,
                    &formatted_buf,
                    formatted_buf.len,
                );
                if (next_len < 0) {
                    try stdout.print("Error: Failed to apply chat template\n", .{});
                    try stdout.flush();
                    break;
                }
                prev_len = @intCast(c.llama_chat_apply_template(
                    tmpl,
                    messages.items.ptr,
                    messages.items.len - 1,
                    false,
                    null,
                    0,
                ));
                prompt = formatted_buf[prev_len..@intCast(next_len)];

                try stdout.print("\n", .{});
                try stdout.flush();
            } else {
                // No tool call - done with this turn
                const response_content = try allocator.dupeZ(u8, result.text);
                allocator.free(result.text);
                try messages.append(allocator, .{
                    .role = "assistant",
                    .content = response_content.ptr,
                });
                break;
            }
        }

        try stdout.print("\n", .{});
        try stdout.flush();

        // Update prev_len for next user input
        const len = c.llama_chat_apply_template(
            tmpl,
            messages.items.ptr,
            messages.items.len,
            false,
            null,
            0,
        );
        if (len >= 0) {
            prev_len = @intCast(len);
        }
    }

    // Free history entries
    for (history.items) |entry| {
        allocator.free(entry);
    }
    history.deinit(allocator);

    try stdout.print("\nGoodbye!\n", .{});
    try stdout.flush();
}

const GenerateResult = struct {
    text: []u8,
    prompt_tokens: usize,
    gen_tokens: usize,
    prompt_time_ns: u64,
    gen_time_ns: u64,
};

const ThinkingState = enum {
    normal,
    maybe_open,    // saw '<', looking for 'think>'
    thinking,      // inside <think>...</think>
    maybe_close,   // saw '<', looking for '/think>'
};

fn drawThinkingBox(writer: anytype, frame_idx: usize) !void {
    const frame = spinner_frames[frame_idx % spinner_frames.len];
    // Clear line and draw box
    try writer.print("\r\x1b[K\x1b[90m┌─────────────────┐\x1b[0m\r\n", .{});
    try writer.print("\x1b[90m│\x1b[0m \x1b[36m{s} thinking...\x1b[0m   \x1b[90m│\x1b[0m\r\n", .{frame});
    try writer.print("\x1b[90m└─────────────────┘\x1b[0m", .{});
    // Move cursor back up 2 lines to overwrite on next frame
    try writer.print("\x1b[2A", .{});
    try writer.flush();
}

fn clearThinkingBox(writer: anytype) !void {
    // Clear the 3 lines of the box
    try writer.print("\r\x1b[K\n\x1b[K\n\x1b[K\x1b[2A\r", .{});
    try writer.flush();
}

// Returns expected UTF-8 sequence length from first byte, 0 if invalid/ASCII
fn utf8SeqLen(first_byte: u8) usize {
    if (first_byte & 0x80 == 0) return 1;        // ASCII
    if (first_byte & 0xE0 == 0xC0) return 2;     // 110xxxxx
    if (first_byte & 0xF0 == 0xE0) return 3;     // 1110xxxx
    if (first_byte & 0xF8 == 0xF0) return 4;     // 11110xxx
    return 1; // Invalid, treat as single byte
}

// Check if byte is a UTF-8 continuation byte (10xxxxxx)
fn isUtf8Continuation(byte: u8) bool {
    return (byte & 0xC0) == 0x80;
}

fn generate(
    allocator: std.mem.Allocator,
    ctx: *c.llama_context,
    vocab: *const c.llama_vocab,
    smpl: *c.llama_sampler,
    prompt: []const u8,
    writer: anytype,
) !GenerateResult {
    var response: std.ArrayList(u8) = .empty;
    errdefer response.deinit(allocator);

    var gen_tokens: usize = 0;
    var prompt_time_ns: u64 = 0;
    var gen_time_ns: u64 = 0;

    // Thinking state machine
    var state: ThinkingState = .normal;
    var tag_buffer: [16]u8 = undefined;
    var tag_len: usize = 0;
    var spinner_frame: usize = 0;
    var last_spinner_update: i64 = 0;
    var thinking_box_shown = false;

    // UTF-8 buffering for incomplete sequences
    var utf8_buffer: [4]u8 = undefined;
    var utf8_len: usize = 0;

    // Check if this is the first generation
    const memory = c.llama_get_memory(ctx);
    const is_first = c.llama_memory_seq_pos_max(memory, 0) == -1;

    // Tokenize the prompt
    const prompt_z = try allocator.dupeZ(u8, prompt);
    defer allocator.free(prompt_z);

    const n_prompt_tokens = -c.llama_tokenize(
        vocab,
        prompt_z.ptr,
        @intCast(prompt.len),
        null,
        0,
        is_first,
        true,
    );

    if (n_prompt_tokens <= 0) {
        return error.TokenizationFailed;
    }

    const tokens = try allocator.alloc(c.llama_token, @intCast(n_prompt_tokens));
    defer allocator.free(tokens);

    const tokenized = c.llama_tokenize(
        vocab,
        prompt_z.ptr,
        @intCast(prompt.len),
        tokens.ptr,
        @intCast(tokens.len),
        is_first,
        true,
    );

    if (tokenized < 0) {
        return error.TokenizationFailed;
    }

    // Create batch for prompt
    var batch = c.llama_batch_get_one(tokens.ptr, @intCast(tokens.len));
    const n_prompt_toks = tokens.len;
    var is_prompt_phase = true;
    var timer = std.time.Timer.start() catch unreachable;

    // Generation loop
    while (true) {
        // Check context space
        const n_ctx = c.llama_n_ctx(ctx);
        const n_ctx_used: u32 = @intCast(c.llama_memory_seq_pos_max(memory, 0) + 1);
        const batch_tokens: u32 = @intCast(batch.n_tokens);
        if (n_ctx_used + batch_tokens > n_ctx) {
            if (thinking_box_shown) {
                try clearThinkingBox(writer);
            }
            try writer.print("\n[Context limit reached]\n", .{});
            try writer.flush();
            break;
        }

        // Decode
        const ret = c.llama_decode(ctx, batch);
        if (ret != 0) {
            return error.DecodeFailed;
        }

        // Record prompt processing time after first decode
        if (is_prompt_phase) {
            prompt_time_ns = timer.read();
            timer.reset();
            is_prompt_phase = false;
        }

        // Sample next token
        const new_token = c.llama_sampler_sample(smpl, ctx, -1);

        // Check for end of generation
        if (c.llama_vocab_is_eog(vocab, new_token)) {
            gen_time_ns = timer.read();
            if (thinking_box_shown) {
                try clearThinkingBox(writer);
            }
            break;
        }

        // Check for Ctrl+C interrupt
        if (generation_interrupted) {
            gen_time_ns = timer.read();
            if (thinking_box_shown) {
                try clearThinkingBox(writer);
            }
            try writer.print("\n\x1b[90m[interrupted]\x1b[0m", .{});
            try writer.flush();
            break;
        }

        gen_tokens += 1;

        // Convert token to text
        var buf: [256]u8 = undefined;
        const n = c.llama_token_to_piece(vocab, new_token, &buf, buf.len, 0, true);
        if (n < 0) {
            return error.TokenConversionFailed;
        }

        const piece = buf[0..@intCast(n)];

        // Process piece through state machine (preserving UTF-8 by writing slices)
        var piece_start: usize = 0;
        var idx: usize = 0;
        while (idx < piece.len) : (idx += 1) {
            const char = piece[idx];
            switch (state) {
                .normal => {
                    if (char == '<') {
                        // Output everything before the '<'
                        if (idx > piece_start) {
                            const slice = piece[piece_start..idx];
                            try writer.print("\x1b[33m", .{});
                            try writer.writeAll(slice);
                            try writer.flush();
                            try response.appendSlice(allocator, slice);
                        }
                        state = .maybe_open;
                        tag_len = 0;
                        tag_buffer[tag_len] = char;
                        tag_len += 1;
                        piece_start = idx + 1;
                    }
                },
                .maybe_open => {
                    tag_buffer[tag_len] = char;
                    tag_len += 1;

                    const target = "<think>";
                    if (tag_len <= target.len and tag_buffer[tag_len - 1] == target[tag_len - 1]) {
                        if (tag_len == target.len) {
                            // Complete <think> tag found
                            state = .thinking;
                            tag_len = 0;
                            piece_start = idx + 1;
                            // Show thinking box
                            try drawThinkingBox(writer, spinner_frame);
                            thinking_box_shown = true;
                            last_spinner_update = std.time.milliTimestamp();
                        }
                    } else {
                        // Not a <think> tag, output buffered tag chars
                        try writer.print("\x1b[33m", .{});
                        try writer.writeAll(tag_buffer[0..tag_len]);
                        try writer.flush();
                        try response.appendSlice(allocator, tag_buffer[0..tag_len]);
                        state = .normal;
                        tag_len = 0;
                        piece_start = idx + 1;
                    }
                },
                .thinking => {
                    // Update spinner animation
                    const now = std.time.milliTimestamp();
                    if (now - last_spinner_update > 80) {
                        spinner_frame += 1;
                        try drawThinkingBox(writer, spinner_frame);
                        last_spinner_update = now;
                    }

                    if (char == '<') {
                        state = .maybe_close;
                        tag_len = 0;
                        tag_buffer[tag_len] = char;
                        tag_len += 1;
                    }
                    piece_start = idx + 1;
                },
                .maybe_close => {
                    tag_buffer[tag_len] = char;
                    tag_len += 1;

                    const target = "</think>";
                    if (tag_len <= target.len and tag_buffer[tag_len - 1] == target[tag_len - 1]) {
                        if (tag_len == target.len) {
                            // Complete </think> tag found
                            state = .normal;
                            tag_len = 0;
                            piece_start = idx + 1;
                            // Clear thinking box
                            try clearThinkingBox(writer);
                            thinking_box_shown = false;
                        }
                    } else {
                        // Not a </think> tag, stay in thinking mode
                        state = .thinking;
                        tag_len = 0;
                        piece_start = idx + 1;
                    }
                },
            }
        }
        // Output remaining content in normal state with UTF-8 buffering
        if (state == .normal and piece_start < piece.len) {
            const slice = piece[piece_start..];

            // Combine with any buffered UTF-8 bytes
            var combined: [260]u8 = undefined;
            var combined_len: usize = 0;

            // Copy buffered bytes first
            for (utf8_buffer[0..utf8_len]) |b| {
                combined[combined_len] = b;
                combined_len += 1;
            }
            utf8_len = 0;

            // Copy new slice
            for (slice) |b| {
                combined[combined_len] = b;
                combined_len += 1;
            }

            // Find where complete UTF-8 sequences end
            var output_end: usize = 0;
            var pos: usize = 0;
            while (pos < combined_len) {
                const seq_len = utf8SeqLen(combined[pos]);
                if (pos + seq_len <= combined_len) {
                    // Complete sequence
                    output_end = pos + seq_len;
                    pos += seq_len;
                } else {
                    // Incomplete sequence at end, buffer it
                    break;
                }
            }

            // Output complete portion
            if (output_end > 0) {
                try writer.print("\x1b[33m", .{});
                try writer.writeAll(combined[0..output_end]);
                try writer.flush();
                try response.appendSlice(allocator, combined[0..output_end]);
            }

            // Buffer incomplete trailing bytes
            if (output_end < combined_len) {
                for (combined[output_end..combined_len]) |b| {
                    utf8_buffer[utf8_len] = b;
                    utf8_len += 1;
                }
            }
        }

        // Prepare next batch
        var single_token = [_]c.llama_token{new_token};
        batch = c.llama_batch_get_one(&single_token, 1);
    }

    // Flush any remaining buffered UTF-8 bytes
    if (utf8_len > 0) {
        try writer.print("\x1b[33m", .{});
        try writer.writeAll(utf8_buffer[0..utf8_len]);
        try response.appendSlice(allocator, utf8_buffer[0..utf8_len]);
    }

    // Final newline after response
    try writer.print("\x1b[0m\n", .{});
    try writer.flush();

    return .{
        .text = try response.toOwnedSlice(allocator),
        .prompt_tokens = n_prompt_toks,
        .gen_tokens = gen_tokens,
        .prompt_time_ns = prompt_time_ns,
        .gen_time_ns = gen_time_ns,
    };
}
