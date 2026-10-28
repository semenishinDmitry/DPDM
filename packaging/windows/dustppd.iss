; Inno Setup script for the Windows installer (per-user, no administrator rights).
; Built in CI:  iscc /DAppVersion=0.1.0 /DVersionNumeric=0.1.0 /DSourceDir=<repo>\dist\dustppd packaging\windows\dustppd.iss

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef VersionNumeric
  #define VersionNumeric "0.0.0"
#endif
#ifndef SourceDir
  #define SourceDir "..\..\dist\dustppd"
#endif

[Setup]
AppId={{6C2B8E0A-6F3E-4E57-9C61-2D3B7E0F5A11}
AppName=dustppd
AppVersion={#AppVersion}
VersionInfoVersion={#VersionNumeric}
AppPublisher=dustppd
AppPublisherURL=https://github.com/semenishinDmitry/DPDM
DefaultDirName={localappdata}\Programs\dustppd
DefaultGroupName=dustppd
PrivilegesRequired=lowest
DisableProgramGroupPage=yes
OutputBaseFilename=dustppd-{#AppVersion}-windows-x64-setup
OutputDir=..\..\dist
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\dustppd.exe
SetupIconFile=icon.ico

[Languages]
Name: "ru"; MessagesFile: "compiler:Languages\Russian.isl"
Name: "en"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{group}\dustppd"; Filename: "{app}\dustppd.exe"
Name: "{group}\{cm:UninstallProgram,dustppd}"; Filename: "{uninstallexe}"
Name: "{userdesktop}\dustppd"; Filename: "{app}\dustppd.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\dustppd.exe"; Description: "{cm:LaunchProgram,dustppd}"; Flags: nowait postinstall skipifsilent
