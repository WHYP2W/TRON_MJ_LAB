[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$Repository = 'https://github.com/limxdynamics/tron2-robot-description.git'
$Revision = 'f547f5bc949f2a4c98e076e61cf6d3ca73d179a0'
$ProjectRoot = Split-Path $PSScriptRoot -Parent
$Destination = Join-Path $ProjectRoot 'assets\robot-description'

if (Test-Path -LiteralPath $Destination) {
    if (-not (Test-Path -LiteralPath (Join-Path $Destination '.git'))) {
        throw "Existing destination is not a Git checkout: $Destination"
    }
    $CurrentRevision = & git --no-optional-locks -C $Destination rev-parse HEAD
    if ($LASTEXITCODE -ne 0 -or $CurrentRevision -ne $Revision) {
        throw "Expected asset revision $Revision. Existing files were not changed."
    }
    $Changes = & git --no-optional-locks -C $Destination status --porcelain --untracked-files=all
    if ($LASTEXITCODE -ne 0 -or $Changes) {
        throw 'Asset checkout has local changes; refusing to overwrite them.'
    }
} else {
    & git -c core.autocrlf=false clone --depth 1 --filter=blob:none --sparse $Repository $Destination
    if ($LASTEXITCODE -ne 0) { throw 'Asset clone failed.' }
    & git -C $Destination fetch --depth 1 origin $Revision
    if ($LASTEXITCODE -ne 0) { throw 'Fetching the pinned asset revision failed.' }
    & git -C $Destination checkout --detach $Revision
    if ($LASTEXITCODE -ne 0) { throw 'Selecting the pinned asset revision failed.' }
}

& git -C $Destination sparse-checkout set tron2a/SFYG_TRON2A/xml tron2a/SFYG_TRON2A/meshes
if ($LASTEXITCODE -ne 0) { throw 'Downloading the SFYG asset files failed.' }

$Model = Join-Path $Destination 'tron2a\SFYG_TRON2A\xml\robot.xml'
$Mesh = Join-Path $Destination 'tron2a\SFYG_TRON2A\meshes\base_Link.STL'
if (-not (Test-Path -LiteralPath $Model) -or -not (Test-Path -LiteralPath $Mesh)) {
    throw "Missing SFYG XML or meshes in $Destination"
}
Write-Output "SFYG assets verified at $Revision"
Write-Output $Destination
