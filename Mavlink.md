```
<protocol>://[bind_host][:bind_port]@[remote_host][:remote_port]
```

## Direct to flight controller
fcu_url:=serial:///dev/ttyUSB0:57600

## One-way listen only {kinda...}
Listen on 0.0.0.0:14550 [defaulted if not present]
Send to 127.0.0.1:14550

```
fcu_url:=udp://@:14550
```

## Bind and send in this case to SITL
Listen on 14555 and send on 14551
```
fcu_url:=udp://:14555@127.0.0.1:14551
```

## Send to a vehicle
192.168.4.1 listening on 14550
```
fcu_url:=udp://@192.168.4.1:14550
```