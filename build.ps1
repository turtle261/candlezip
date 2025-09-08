$vcvars = 'C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat'
cmd /c "`"$vcvars`" && cargo build --release"
# Also build the rwkvzip binary crate
cmd /c "`"$vcvars`" && cd rwkvzip && cargo build --release"
#This fixes issues with CUDA during compilation and makes CUDA work