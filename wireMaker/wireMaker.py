#!/usr/bin/python3

import argparse
import os

def addClient(template, svrConf='svrWire.conf', keepalive = None):
    srv = svrParse(svrConf)
    cli = cliParse(template)
    nxt = nextIp(svrConf)
    prefix = srv['prefix']

    ## Generate keys
    os.system('umask 077 && wg genkey > t_cprv')
    os.system('cat t_cprv | wg pubkey > t_cpub')
    with open('t_cprv') as f:
        cliPriv = f.read().strip()
    with open('t_cpub') as f:
        cliPub = f.read().strip()
    os.remove('t_cprv')
    os.remove('t_cpub')

    os.system('umask 077 && wg genpsk > t_psk')
    with open('t_psk') as f:
        psk = f.read().strip()
    os.remove('t_psk')

    ## Write new client config
    newCliFile = f'cliWire-{nxt}.conf'
    cliText = f"""[Interface]
PrivateKey = {cliPriv}
Address = {prefix}.{nxt}/24

[Peer]
PublicKey = {cli['svr_pub']}
Endpoint = {cli['endpoint']}
AllowedIPs = {prefix}.0/24
PresharedKey = {psk}
{keepAlive('server', keepalive)}
"""
    with open(newCliFile, 'w') as f:
        f.write(cliText)

    ## Append to server config
    peerBlock = f"""[Peer]
PublicKey = {cliPub}
AllowedIPs = {prefix}.{nxt}/32
PresharedKey = {psk}
{keepAlive('server', keepalive)}
"""
    with open(svrConf, 'a') as f:
        f.write(peerBlock)
    print(f'Added new client: {newCliFile}   (IP {prefix}.{nxt})')


def cliParse(path):
    data = {}
    with open(path) as f:
        for line in f:
            if line.startswith('PublicKey'):
                data['svr_pub'] = line.split('=', 1)[1].strip()
            elif line.startswith('Endpoint'):
                data['endpoint'] = line.split('=', 1)[1].strip()
            elif line.startswith('AllowedIPs'):
                p = line.split('=', 1)[1].strip()
                data['prefix'] = '.'.join(p.split('.')[:3])
    return data


def keepAlive(side, keepalive):
    if keepalive in (side, 'both'):
        return 'PersistentKeepalive = 25\n'
    return ''


def nextIp(svrConf='svrWire.conf'):
    used = []
    with open(svrConf) as f:
        for line in f:
            if line.startswith('AllowedIPs'):
                ip = line.split('=')[1].strip()
                octet = int(ip.split('.')[3].split('/')[0])
                used.append(octet)
    nextOct = max(used) + 1 if used else 2
    if nextOct > 254:
        raise Exception('Subnet full: cannot allocate beyond .254')
    return nextOct


def peerGen(endIp, endPt, svrPub, cliOct, prefix, keepalive = None):
    """Generate and return peer information"""
    os.system('umask 077 && wg genkey > cliPrvkey')
    os.system('cat cliPrvkey | wg pubkey > cliPubkey')
    os.system('umask 077 && wg genpsk > psk 2>/dev/null')
    with open('cliPrvkey') as iFile:
        cliPriv = iFile.read().splitlines()[0]
    with open('cliPubkey') as iFile:
        cliPub = iFile.read().splitlines()[0]
    with open('psk') as iFile:
        psk = iFile.read().splitlines()[0]
    cliIp = f'Address = {prefix}.{cliOct}/24'
    cliQck = f"""[Interface]
PrivateKey = {cliPriv}
{cliIp}

[Peer]
PublicKey = {svrPub}
Endpoint = {endIp}:{endPt}
AllowedIPs = {prefix}.0/24
PresharedKey = {psk}
{keepAlive('client', keepalive)}
"""
    return cliQck, cliPub, psk, cliOct


def templateFind():
    """Select highest-numbered cliWire-N.conf, otherwise cliWire.conf"""
    numbered = []
    for f in os.listdir('.'):
        if f.startswith('cliWire-') and f.endswith('.conf'):
            try:
                num = int(f.replace('cliWire-', '').replace('.conf', ''))
                numbered.append((num, f))
            except:
                continue
    if numbered:
        numbered.sort()
        return numbered[-1][1]
    if os.path.exists('cliWire.conf'):
        return 'cliWire.conf'
    raise Exception('No client config files found.')


def secureConfigs():
    """Ensure all WireGuard-related files have 0600 permissions."""
    files = []

    # Standard single-client and server configs
    files.append('cliWire.conf')
    files.append('svrWire.conf')

    # Any numbered client configs
    for f in os.listdir('.'):
        if f.startswith('cliWire-') and f.endswith('.conf'):
            files.append(f)
    for f in files:
        if os.path.exists(f):
            try:
                os.chmod(f, 0o600)
            except Exception as e:
                print(f'Warning: could not chmod 0600 on {f}: {e}')


def svrParse(path = 'svrWire.conf'):
    data = {}
    with open(path) as f:
        for line in f:
            if line.startswith('PrivateKey'):
                data['svr_priv'] = line.split('=', 1)[1].strip()
            elif line.startswith('ListenPort'):
                data['port'] = line.split('=', 1)[1].strip()
            elif line.startswith('Address'):
                addr = line.split('=', 1)[1].strip()
                data['prefix'] = '.'.join(addr.split('.')[:3])
    return data


def main(endIp, endPt, multiClient = False, prefix = '10.249.177', keepalive = None):
    os.system('umask 077 && wg genkey > svrPrvkey')
    os.system('cat svrPrvkey | wg pubkey > svrPubkey')
    with open('svrPrvkey') as iFile:
        svrPriv = iFile.read().splitlines()[0]
    with open('svrPubkey') as iFile:
        svrPub = iFile.read().splitlines()[0]

    ## Create environment for one client
    if multiClient is False:
        cliQck = peerGen(endIp, endPt, svrPub, 2, prefix, keepalive)
        svrQck = f"""[Interface]
PrivateKey = {svrPriv}
ListenPort = {endPt}
Address = {prefix}.1

[Peer]
PublicKey = {cliQck[1]}
AllowedIPs = {prefix}.2/32
PresharedKey = {cliQck[2]}
{keepAlive('server', keepalive)}
"""

        with open('svrWire.conf', 'w') as oFile:
            oFile.write(f'{svrQck}')
        with open('cliWire.conf', 'w') as oFile:
            oFile.write(f'{cliQck[0]}')

    ## Create environment for multiple clients
    else:
        pList = []
        for i in range (2, multiClient + 1):
            pList.append(peerGen(endIp, endPt, svrPub, i, prefix, keepalive))

        fList = []
        for p in pList:
            cStr = f"""[Peer]
PublicKey = {p[1]}
AllowedIPs = {prefix}.{p[3]}/32
PresharedKey = {p[2]}
{keepAlive('server', keepalive)}
"""
            fList.append(cStr)

        cliStr = '\n'.join(fList)
        svrQck = f"""[Interface]
PrivateKey = {svrPriv}
ListenPort = {endPt}
Address = {prefix}.1

{cliStr}
"""
        with open('svrWire.conf', 'w') as oFile:
            oFile.write(f'{svrQck}')

        ## Generate client configs
        for i in range(len(pList)):
            with open(f'cliWire-{i + 2}.conf', 'w') as oFile:
                oFile.write(f'{pList[i][0]}')

if __name__ == '__main__':

    ## Inputs
    psr = argparse.ArgumentParser(description = 'A wrapper for creating WireGuard configurations')
    psr.add_argument('-k', help = 'Add a keepalive', choices = ['client', 'server', 'both'])
    psr.add_argument('-m', help = 'Multi client mode')
    psr.add_argument('-s', help = 'WireGuard server endpoint IP address')
    psr.add_argument('-p', help = '51820 is the default port')
    psr.add_argument('--add', action='store_true', help='Add client to existing environment')
    psr.add_argument('--prefix', help = 'Specifies first three octets [Default is 10.249.177]')
    args = psr.parse_args()

    ## Handle keepalives
    keepalive = args.k

    ## Parsing
    if args.add:
        template = templateFind()
        addClient(template, keepalive = keepalive)
        secureConfigs()
        exit(0)

    endIp = args.s
    if endIp is None:
        print('-s is required')
        exit(1)

    if args.p is None:
        endPt = 51820
    else:
        endPt = args.p
    
    if args.prefix is None:
        prefix = '10.249.177'
    else:
        prefix = args.prefix

    if args.m is not None:
        x = main(endIp, endPt, int(args.m), prefix, keepalive = keepalive)
    else:
        main(endIp, endPt, prefix = prefix, keepalive = keepalive)

    ## Cleanup
    secureConfigs()
    os.remove('cliPrvkey')
    os.remove('cliPubkey')
    os.remove('svrPrvkey')
    os.remove('svrPubkey')
    os.remove('psk')
