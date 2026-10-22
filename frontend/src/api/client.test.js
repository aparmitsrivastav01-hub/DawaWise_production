import { afterEach, describe, expect, it, vi } from 'vitest';

afterEach(() => {
  vi.unstubAllGlobals();
  vi.unstubAllEnvs();
  vi.resetModules();
});

async function loadClient(baseUrl) {
  vi.stubEnv('VITE_API_BASE_URL', baseUrl);
  vi.resetModules();
  return import('./client.js');
}

function response(payload, ok = true) {
  return { ok, status: ok ? 200 : 500, text: async () => JSON.stringify(payload) };
}

describe('backend API client', () => {
  it('normalizes a base URL that already ends in /api and calls the existing search route', async () => {
    const fetchMock = vi.fn().mockResolvedValue(response({ status: 'not_found' }));
    vi.stubGlobal('fetch', fetchMock);
    const client = await loadClient('https://dawawise-production.onrender.com/api///');

    await client.searchMedicines('cetirizine');

    expect(client.API_BASE_URL).toBe('https://dawawise-production.onrender.com');
    expect(fetchMock).toHaveBeenCalledWith(
      'https://dawawise-production.onrender.com/api/search',
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({ query: 'cetirizine' }),
      }),
    );
  });

  it('uses the configured base URL for the existing health route', async () => {
    const fetchMock = vi.fn().mockResolvedValue(response({ status: 'ok' }));
    vi.stubGlobal('fetch', fetchMock);
    const client = await loadClient('https://dawawise-production.onrender.com/');

    await client.getHealth();

    expect(fetchMock).toHaveBeenCalledWith(
      'https://dawawise-production.onrender.com/api/health',
      expect.objectContaining({ method: 'GET' }),
    );
  });
});
