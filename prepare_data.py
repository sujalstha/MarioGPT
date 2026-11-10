"""Acquire the exact pinned VGLC corpus without storing account credentials."""
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO
REV='0e86a8f31f20ecad4eaa5741ff061af88767f7fb'
root=Path(__file__).parent/'data';root.mkdir(exist_ok=True)
with urlopen(f'https://github.com/TheVGLC/TheVGLC/archive/{REV}.zip') as r:
 z=ZipFile(BytesIO(r.read()))
 for name in z.namelist():
  parts=name.split('/')
  if len(parts)==4 and parts[1] in ['Super Mario Bros','Super Mario Bros 2 (Japan)'] and parts[2]=='Processed' and name.endswith('.txt'):
   (root/parts[-1]).write_bytes(z.read(name))
 license_name=f'TheVGLC-{REV}/License.md'
 (root/'VGLC-License.md').write_bytes(z.read(license_name))
print(f'Prepared {len(list(root.glob("*.txt")))} level maps at {REV}')
