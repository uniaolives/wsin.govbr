# qHTTP Formal Grammar

This document provides the formal syntax for the qHTTP protocol using Augmented Backus-Naur Form (ABNF).

```abnf
; qHTTP Formal Grammar in ABNF (Augmented Backus-Naur Form)
; Based on RFC 5234 and extending HTTP/1.1 semantics

qhttp-message = start-line *( header-field CRLF ) CRLF [ message-body ]

start-line = request-line / status-line

request-line = method SP request-target SP qhttp-version CRLF

status-line = qhttp-version SP status-code SP reason-phrase CRLF

qhttp-version = "qHTTP/" 1*DIGIT "." 1*DIGIT

method = "OBSERVE"
       / "SUPERPOSE"
       / "ENTANGLE"
       / "INTERFERE"
       / "COLLAPSE"
       / "COHERE"
       / "DECOHERE"
       / token

status-code = "207" ; Multi-State (Superposition)
            / "480" ; Decoherence
            / "481" ; Entanglement Broken
            / "482" ; Uncertainty Limit
            / 3DIGIT

reason-phrase = *( HTAB / SP / VCHAR / obs-text )

request-target = authority-form / absolute-form / origin-form / asterisk-form

header-field = field-name ":" OWS field-value OWS

field-name = token
field-value = *( field-content / obs-fold )
field-content = field-vchar [ 1*( SP / HTAB ) field-vchar ]
field-vchar = VCHAR / obs-text

token = 1*tchar
tchar = "!" / "#" / "$" / "%" / "&" / "'" / "*" / "+" / "-" / "." / "^" / "_" / "`" / "|" / "~" / DIGIT / ALPHA

message-body = *OCTET

; Quantum specific header value structures
probability-amplitude = amplitude-pair *( "," OWS amplitude-pair )
amplitude-pair = state-id "=" complex-number
state-id = token
complex-number = float [ ( "+" / "-" ) float "j" ]
float = 1*DIGIT [ "." 1*DIGIT ]

coherence-time = 1*DIGIT ; duration in milliseconds

entanglement-id = token

observer-id = token

; ABNF Core Rules & HTTP/1.1 Fragments
origin-form    = absolute-path [ "?" query ]
absolute-form  = absolute-URI
authority-form = authority
asterisk-form  = "*"

absolute-path = 1*( "/" segment )
segment       = *pchar
pchar         = unreserved / pct-encoded / sub-delims / ":" / "@"
query         = *( pchar / "/" / "?" )
absolute-URI  = scheme ":" hier-part [ "?" query ]
scheme        = "quantum" / ALPHA *( ALPHA / DIGIT / "+" / "-" / "." )
hier-part     = "//" authority path-abempty / path-absolute / path-rootless / path-empty
authority     = [ userinfo "@" ] host [ ":" port ]
path-abempty  = *( "/" segment )
path-absolute = "/" [ segment-nz *( "/" segment ) ]
path-rootless = segment-nz *( "/" segment )
path-empty    = 0pchar
userinfo      = *( unreserved / pct-encoded / sub-delims / ":" )
host          = IP-literal / IPv4address / reg-name
port          = *DIGIT
IP-literal    = "[" ( IPv6address / IPvFuture  ) "]"
IPvFuture     = "v" 1*HEXDIG "." 1*( unreserved / sub-delims / ":" )
unreserved    = ALPHA / DIGIT / "-" / "." / "_" / "~"
pct-encoded   = "%" HEXDIG HEXDIG
sub-delims    = "!" / "$" / "&" / "'" / "(" / ")" / "*" / "+" / "," / ";" / "="

OWS = *( SP / HTAB )
CRLF = %x0D %x0A
SP = %x20
HTAB = %x09
VCHAR = %x21-7E
DIGIT = %x30-39
ALPHA = %x41-5A / %x61-7A
HEXDIG = DIGIT / "A" / "B" / "C" / "D" / "E" / "F"
OCTET = %x00-FF
obs-text = %x80-FF
obs-fold = CRLF 1*( SP / HTAB )
```
