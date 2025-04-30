let
  pkgs = import <nixpkgs> {system = "x86_64-linux";};
in
  pkgs.mkShell {
    buildInputs = with pkgs; [
      ruff
      (python3.withPackages (ps:
        with ps; [
          pandas
          scikit-learn
          transformers
          torch
          datasets
          evaluate
          accelerate
        ]))
    ];
  }
