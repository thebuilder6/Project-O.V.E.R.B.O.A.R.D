; Inno Setup script for the Windows desktop build.
;
;   pyinstaller --noconfirm overboard.spec
;   iscc /DVersion=1.2.3 installer.iss
;
; A per-user install (no admin prompt): the app already keeps its missions and
; calibration under %LOCALAPPDATA%\OVERBOARD.

#ifndef Version
#define Version "dev"
#endif

[Setup]
AppId={{6C1F3B7E-2D34-4C8A-9E51-7B0E2F5A9C11}
AppName=OVERBOARD
AppVersion={#Version}
DefaultDirName={localappdata}\OVERBOARD
DefaultGroupName=OVERBOARD
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=dist
OutputBaseFilename=OVERBOARD-setup-{#Version}
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
UninstallDisplayIcon={app}\OVERBOARD.exe

[Tasks]
Name: "desktopicon"; Description: "Create a desktop shortcut"; GroupDescription: "Additional icons:"

[Files]
Source: "dist\OVERBOARD\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\OVERBOARD"; Filename: "{app}\OVERBOARD.exe"
Name: "{autodesktop}\OVERBOARD"; Filename: "{app}\OVERBOARD.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\OVERBOARD.exe"; Description: "Start the studio"; Flags: nowait postinstall skipifsilent
