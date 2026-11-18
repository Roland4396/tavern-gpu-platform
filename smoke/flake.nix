{
  description = "DSW GPU node Nix and host-driver smoke test";

  outputs = { self }:
    let
      system = "x86_64-linux";
      smoke = derivation {
        inherit system;
        name = "gpu-platform-smoke";
        builder = "/bin/sh";
        args = [
          "-c"
          ''
            /bin/mkdir -p "$out/bin"
            /bin/cp ${./gpu-platform-smoke} "$out/bin/gpu-platform-smoke"
            /bin/chmod 0555 "$out/bin/gpu-platform-smoke"
          ''
        ];
      };
    in {
      packages.${system} = {
        default = smoke;
        gpu-platform-smoke = smoke;
      };
    };
}
