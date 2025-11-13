sudo growpart /dev/mmcblk0 2

sudo resize2fs /dev/mmcblk0p2


sudo tee /usr/local/sbin/expand-rootfs.sh >/dev/null <<'EOF'
#!/bin/bash
set -e
DISK=/dev/mmcblk0
PART=2
echo "Expanding root partition..."
growpart $DISK $PART
echo "Resizing filesystem..."
resize2fs ${DISK}p${PART}
echo "Root filesystem expanded successfully."
EOF

sudo chmod +x /usr/local/sbin/expand-rootfs.sh
