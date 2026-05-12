// modified from https://github.com/kairi003/Get-cookies.txt-LOCALLY/blob/0e9952ccb1b1d914c3dc42fa496219bb7285e2dc/src/modules/cookie_format.mjs
jsonToNetscapeMapper = (cookies) => {
  const header = [
    '# Netscape HTTP Cookie File',
    '# https://curl.haxx.se/rfc/cookie_spec.html',
    '# This is a generated file! Do not edit.',
    ''
  ]
  const lines = [...header, ...cookies.map(
    ({ domain, expirationDate, path, secure, name, value }) => {
      const includeSubDomain = !!domain?.startsWith('.');
      const expiry = expirationDate?.toFixed() ?? '0';
      const arr = [domain, includeSubDomain, path, secure, expiry, name, value];
      return arr.map((v) =>
        typeof v === 'boolean' ? v.toString().toUpperCase() : v,
      );
    },
  ).map((row) => row.join('\t'))]
  return lines.join('\n');
};

cookies = await cookieStore.getAll().then(jsonToNetscapeMapper);
cookies
