import os
import shutil

src_dir = 'path'
dest_dir = 'path'

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)

for cls in os.listdir(src_dir):
    src_cls = os.path.join(src_dir, cls)
    dest_cls = os.path.join(dest_dir, cls)

    for view in os.listdir(src_cls):
        src_view = os.path.join(src_cls, str(view))
        dest_view = os.path.join(dest_cls, str(view))
        if not os.path.exists(dest_view):
            os.makedirs(dest_view)
        for image in os.listdir(src_view):
            src_path = os.path.join(src_view, image)
            dest_path = os.path.join(dest_view, image)
            shutil.copy2(src_path, dest_path)

print(f"Copied path: {dest_dir}")