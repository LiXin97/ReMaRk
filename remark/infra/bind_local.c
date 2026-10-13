// LD_PRELOAD shim: rewrite wildcard binds (0.0.0.0 / ::) to loopback (127.0.0.1 / ::1).
// Needed on some GPU-node containers, where binding the wildcard address fails with EADDRINUSE on any port,
// which breaks torch.distributed's TCPStore. Single-node training only (all ranks talk over loopback).
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <netinet/in.h>
#include <string.h>
#include <sys/socket.h>

typedef int (*bind_fn)(int, const struct sockaddr *, socklen_t);

int bind(int fd, const struct sockaddr *addr, socklen_t len) {
    static bind_fn real_bind = 0;
    if (!real_bind) real_bind = (bind_fn)dlsym(RTLD_NEXT, "bind");
    if (addr && addr->sa_family == AF_INET && len >= sizeof(struct sockaddr_in)) {
        struct sockaddr_in a4;
        memcpy(&a4, addr, sizeof(a4));
        if (a4.sin_addr.s_addr == htonl(INADDR_ANY)) {
            a4.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
            return real_bind(fd, (const struct sockaddr *)&a4, sizeof(a4));
        }
    } else if (addr && addr->sa_family == AF_INET6 && len >= sizeof(struct sockaddr_in6)) {
        struct sockaddr_in6 a6;
        memcpy(&a6, addr, sizeof(a6));
        if (memcmp(&a6.sin6_addr, &in6addr_any, sizeof(struct in6_addr)) == 0) {
            // Fail IPv6 wildcard binds so callers fall back to IPv4 (-> 127.0.0.1). Rebinding to ::1 would
            // leave IPv4 clients connecting to 127.0.0.1 with no listener (observed: TCPStore hang).
            errno = EADDRNOTAVAIL;
            return -1;
        }
    }
    return real_bind(fd, addr, len);
}
