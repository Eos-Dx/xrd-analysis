[Version]
Class=IEXPRESS
SEDVersion=3

[Options]
PackagePurpose=InstallApp
ShowInstallProgramWindow=1
HideExtractAnimation=1
UseLongFileName=1
InsideCompressed=0
CAB_FixedSize=0
CAB_ResvCodeSigning=0
RebootMode=I
TargetName=C:\dev\xrd-analysis\src\hardware\bin\run_d2xc.exe
FriendlyName=D2XC Software Launcher
AppLaunched=run_d2xc.bat
PostInstallCmd=<None>
AdminQuietInstCmd=
UserQuietInstCmd=
SourceFiles=SourceFiles
SelfDelete=0

[SourceFiles]
SourceFiles0=C:\dev\xrd-analysis\src\hardware\bin

[SourceFiles0]
%FILE0%=run_d2xc.bat

[Strings]
FILE0=run_d2xc.bat
