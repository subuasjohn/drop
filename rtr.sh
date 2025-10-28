#!/bin/bash

# ip route add default via 192.168.1.1

## $wanNIC for WAN and $lanNic for LAN
wanNIC='wlan0'
lanNIC='eth0'

sudo sysctl -w net.ipv4.ip_forward=1

# Flush old rules
sudo iptables -F
sudo iptables -t nat -F

# Default policies
sudo iptables -P INPUT ACCEPT
sudo iptables -P FORWARD ACCEPT
sudo iptables -P OUTPUT ACCEPT

# NAT (masquerade) so LAN traffic can go out over $wanNIC
sudo iptables -t nat -A POSTROUTING -o $wanNIC -j MASQUERADE

# Allow forwarding from $lanNic -> $wanNIC
sudo iptables -A FORWARD -i $lanNIC -o $wanNIC -j ACCEPT
sudo iptables -A FORWARD -i $wanNIC -o $lanNIC -m state --state RELATED,ESTABLISHED -j ACCEPT
