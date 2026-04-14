const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const exe = b.addExecutable(.{
        .name = "basi-cli",
        .root_module = b.createModule(.{
            .root_source_file = b.path("src/main.zig"),
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        }),
    });

    // Include paths for llama.cpp headers
    exe.root_module.addIncludePath(.{ .cwd_relative = "/home/alberto/llama.cpp/include" });
    exe.root_module.addIncludePath(.{ .cwd_relative = "/home/alberto/llama.cpp/ggml/include" });

    // Library search path
    exe.root_module.addLibraryPath(.{ .cwd_relative = "/home/alberto/llama.cpp/build_vulkan/bin" });

    // Link llama.cpp libraries
    exe.root_module.linkSystemLibrary("llama", .{});
    exe.root_module.linkSystemLibrary("ggml", .{});
    exe.root_module.linkSystemLibrary("ggml-base", .{});

    // Add rpath so the executable can find libraries at runtime
    exe.root_module.addRPath(.{ .cwd_relative = "/home/alberto/llama.cpp/build_vulkan/bin" });

    b.installArtifact(exe);

    // Run step
    const run_step = b.step("run", "Run BASI-CLI");
    const run_cmd = b.addRunArtifact(exe);
    run_step.dependOn(&run_cmd.step);
    run_cmd.step.dependOn(b.getInstallStep());

    if (b.args) |args| {
        run_cmd.addArgs(args);
    }
}
