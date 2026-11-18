{
  description = "Vistiloria unified GPU runtime profiles";

  outputs = { self }:
    let
      system = "x86_64-linux";
      stack = builtins.fromJSON (builtins.readFile ./stack.json);
      workloadNames = builtins.attrNames stack.workloads;
      componentsFile = ./components.json;
      componentsAvailable = builtins.pathExists componentsFile;
      components = if componentsAvailable then builtins.fromJSON (builtins.readFile componentsFile) else {};
      componentNames = if componentsAvailable then builtins.attrNames components.components else [];
      componentByPath = path:
        let
          matches = builtins.filter (
            name:
              let value = components.components.${name};
              in (value.asset_path or null) == path && value ? store_path
          ) componentNames;
        in
          if builtins.length matches == 1
          then builtins.storePath components.components.${builtins.head matches}.store_path
          else throw "expected exactly one component for ${path}, got ${toString (builtins.length matches)}";

      mkContractProfile = name: derivation {
        inherit system;
        name = "gpu-profile-${name}";
        builder = "/bin/sh";
        args = [
          "-c"
          ''
            /bin/mkdir -p "$out/bin" "$out/share/gpu-runtime"
            /bin/cp ${./stack.json} "$out/share/gpu-runtime/stack.json"
            /bin/cp ${./profile_guard.py} "$out/share/gpu-runtime/profile_guard.py"
            /bin/cp ${./profile-runner} "$out/bin/gpu-profile-${name}"
            /bin/sed -i 's/@WORKLOAD@/${name}/g' "$out/bin/gpu-profile-${name}"
            /bin/chmod 0555 "$out/bin/gpu-profile-${name}"
          ''
        ];
      };

      mkActualProfile = name:
        let
          runtimeCore = componentByPath "/home/worker/miniconda3/envs/runtime-core";
          qwenEnv = componentByPath "/home/worker/.venvs/qwen38-sglang-0.5.17";
          graphEnv = componentByPath "/home/worker/.venvs/deep-learning";
          h3Env = componentByPath "/home/worker/.venvs/sglang-h3";
          h3Source = componentByPath "/home/worker/src/sglang-h3";
          ttsEnv = componentByPath "/home/worker/codes/audio-synthesis/.venv";
          graphSource = componentByPath "/home/worker/.local/share/graph-stack/core";
          ttsSource = componentByPath "/home/worker/codes/audio-synthesis";
          gpuRuntimeSource = componentByPath "/home/worker/.local/libexec/gpu-runtime";
          launcher = {
            glm = "${gpuRuntimeSource}/glm_launcher.py";
            deepseek = "${gpuRuntimeSource}/deepseek_launcher.py";
            qwen38_native = "${gpuRuntimeSource}/qwen38_launcher.py";
            qwen38_yarn = "${gpuRuntimeSource}/qwen38_launcher.py";
            graph = "${gpuRuntimeSource}/graph_launcher.py";
            tts = null;
            h3 = "${gpuRuntimeSource}/h3_launcher.py";
            h3_ref = "${gpuRuntimeSource}/h3_launcher.py";
            seedvr3 = "${gpuRuntimeSource}/seedvr3_service.py";
          }.${name};
          runner =
            if name == "glm" then ''
              #! /bin/sh
              set -eu
              DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
              export GLM_PYTHON=${runtimeCore}/bin/python
              exec /usr/bin/python3 "$DIR/lib/glm_launcher.py" "$@"
            '' else if name == "deepseek" then ''
              #! /bin/sh
              set -eu
              DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
              export DEEPSEEK_PYTHON=${runtimeCore}/bin/python
              exec /usr/bin/python3 "$DIR/lib/deepseek_launcher.py" "$@"
            '' else if name == "qwen38_native" || name == "qwen38_yarn" then ''
              #! /bin/sh
              set -eu
              DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
              export QWEN38_PYTHON=${qwenEnv}/bin/python
              exec /usr/bin/python3 "$DIR/lib/qwen38_launcher.py" "$@"
            '' else if name == "graph" then ''
              #! /bin/sh
              set -eu
              DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
              export GRAPH_REPO=${graphSource}
              export GRAPH_PYTHON=${graphEnv}/bin/python
              export PYTHONPATH="$DIR/lib:${graphSource}''${PYTHONPATH:+:$PYTHONPATH}"
              exec ${graphEnv}/bin/python "$DIR/lib/graph_launcher.py" "$@"
            '' else if name == "tts" then ''
              #! /bin/sh
              set -eu
              DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
              export TTS_ENGINE_ROOT=${ttsSource}
              export PYTHONPATH="${ttsSource}''${PYTHONPATH:+:$PYTHONPATH}"
              exec ${ttsEnv}/bin/python -m tts_runtime.launcher "$@"
            '' else if name == "h3" || name == "h3_ref" then ''
              #! /bin/sh
              set -eu
              DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
              export H3_PYTHON=${h3Env}/bin/python
              export PYTHONPATH="${h3Source}/python''${PYTHONPATH:+:$PYTHONPATH}"
              exec /usr/bin/python3 "$DIR/lib/h3_launcher.py" "$@"
            '' else if name == "seedvr3" then ''
              #! /bin/sh
              set -eu
              DIR=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
              export GRAPH_REPO=${graphSource}
              export GRAPH_PYTHON=${graphEnv}/bin/python
              export GRAPH_LAUNCHER="$DIR/lib/graph_launcher.py"
              export PYTHONPATH="$DIR/lib:${graphSource}''${PYTHONPATH:+:$PYTHONPATH}"
              exec ${graphEnv}/bin/python "$DIR/lib/seedvr3_service.py" "$@"
            '' else throw "no actual runner for workload ${name}";
          runnerFile = builtins.toFile "gpu-run-${name}" runner;
          launcherCopy = if launcher == null then "" else ''
            /bin/cp ${launcher} "$out/lib/${name}-launcher.py"
          '';
          extraFiles =
            if name == "seedvr3" then ''
              /bin/cp ${gpuRuntimeSource}/seedvr3_benchmark.py "$out/lib/seedvr3_benchmark.py"
              /bin/cp ${gpuRuntimeSource}/graph_launcher.py "$out/lib/graph_launcher.py"
            '' else if name == "graph" then ''
              /bin/cp ${gpuRuntimeSource}/graph_launcher.py "$out/lib/graph_launcher.py"
            '' else "";
          renameLauncher =
            if name == "glm" then ''/bin/mv "$out/lib/glm-launcher.py" "$out/lib/glm_launcher.py"''
            else if name == "deepseek" then ''/bin/mv "$out/lib/deepseek-launcher.py" "$out/lib/deepseek_launcher.py"''
            else if name == "qwen38_native" || name == "qwen38_yarn" then ''/bin/mv "$out/lib/${name}-launcher.py" "$out/lib/qwen38_launcher.py"''
            else if name == "h3" || name == "h3_ref" then ''/bin/mv "$out/lib/${name}-launcher.py" "$out/lib/h3_launcher.py"''
            else if name == "seedvr3" then ''/bin/mv "$out/lib/seedvr3-launcher.py" "$out/lib/seedvr3_service.py"''
            else "";
        in derivation {
          inherit system;
          name = "gpu-profile-${name}";
          builder = "/bin/sh";
          args = [
            "-c"
            ''
              /bin/mkdir -p "$out/bin" "$out/lib" "$out/share/gpu-runtime"
              /bin/cp ${./stack.json} "$out/share/gpu-runtime/stack.json"
              /bin/cp ${./profile_guard.py} "$out/share/gpu-runtime/profile_guard.py"
              /bin/cp ${./profile-runner} "$out/bin/gpu-profile-${name}"
              /bin/sed -i 's/@WORKLOAD@/${name}/g' "$out/bin/gpu-profile-${name}"
              /bin/cp ${runnerFile} "$out/bin/gpu-run-${name}"
              /bin/chmod 0555 "$out/bin/gpu-profile-${name}" "$out/bin/gpu-run-${name}"
              ${launcherCopy}
              ${extraFiles}
              ${renameLauncher}
              if /bin/ls "$out/lib"/*.py >/dev/null 2>&1; then
                /bin/chmod 0555 "$out/lib"/*.py
              fi
            ''
          ];
        };

      mkProfile = name: if componentsAvailable then mkActualProfile name else mkContractProfile name;
      profiles = builtins.listToAttrs (map (name: {
        name = "profile-${name}";
        value = mkProfile name;
      }) workloadNames);

      assetsProfile =
        let
          storedNames = builtins.filter (name: components.components.${name} ? store_path) componentNames;
          links = builtins.concatStringsSep "\n" (map (
            name:
              let path = builtins.storePath components.components.${name}.store_path;
              in ''/bin/ln -s ${path} "$out/share/gpu-runtime/components/${name}"''
          ) storedNames);
        in derivation {
          inherit system;
          name = "gpu-profile-assets";
          builder = "/bin/sh";
          args = [
            "-c"
            ''
              /bin/mkdir -p "$out/share/gpu-runtime/components"
              /bin/cp ${componentsFile} "$out/share/gpu-runtime/components.json"
              ${links}
            ''
          ];
        };

      contract = derivation {
        inherit system;
        name = "gpu-stack-contract";
        builder = "/bin/sh";
        args = [
          "-c"
          ''
            /bin/mkdir -p "$out/share/gpu-runtime"
            /bin/cp ${./stack.json} "$out/share/gpu-runtime/stack.json"
            /bin/cp ${./inventory.py} "$out/share/gpu-runtime/inventory.py"
            /bin/cp ${./compare_inventory.py} "$out/share/gpu-runtime/compare_inventory.py"
            /bin/cp ${./profile_guard.py} "$out/share/gpu-runtime/profile_guard.py"
            /bin/cp ${./model_manifest.py} "$out/share/gpu-runtime/model_manifest.py"
            /bin/cp ${./artifact_sync.py} "$out/share/gpu-runtime/artifact_sync.py"
            /bin/cp ${./artifacts.json} "$out/share/gpu-runtime/artifacts.json"
            /bin/cp ${./activate_profiles.py} "$out/share/gpu-runtime/activate_profiles.py"
            /bin/cp ${./activate_node_config.py} "$out/share/gpu-runtime/activate_node_config.py"
            /bin/cp ${./build_components.py} "$out/share/gpu-runtime/build_components.py"
            /bin/chmod 0444 "$out/share/gpu-runtime"/*
          ''
        ];
      };
    in {
      packages.${system} = profiles // {
        default = contract;
        contract = contract;
        profile-assets = if componentsAvailable then assetsProfile else contract;
      };
      checks.${system} = profiles // {
        contract = contract;
        profile-assets = if componentsAvailable then assetsProfile else contract;
      };
    };
}
