import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import SearchPage from './SearchPage.jsx';
import { searchMedicines } from '../api/client.js';

vi.mock('../api/client.js', () => ({ searchMedicines: vi.fn() }));

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe('live medicine search flow', () => {
  it('sends a generic name and strength to the live search API', async () => {
    searchMedicines.mockResolvedValue({
      status: 'matched',
      match_source: 'tinyfish',
      medicine: { name: 'Metformin 500 mg', verification_status: 'Source checked' },
    });
    const user = userEvent.setup();
    render(<SearchPage />);

    await user.type(screen.getByRole('searchbox', { name: /search medicine/i }), 'metformin 500');
    await user.click(screen.getByRole('button', { name: 'Search', exact: true }));

    await waitFor(() => expect(searchMedicines).toHaveBeenCalledWith('metformin 500'));
    expect(await screen.findByText('Live source results')).toBeTruthy();
  });

  it('sends a brand-specific query to the live search API', async () => {
    searchMedicines.mockResolvedValue({ status: 'not_found', medicine: null });
    const user = userEvent.setup();
    render(<SearchPage />);

    await user.type(screen.getByRole('searchbox', { name: /search medicine/i }), 'Dolo 650');
    await user.click(screen.getByRole('button', { name: 'Search', exact: true }));

    await waitFor(() => expect(searchMedicines).toHaveBeenCalledWith('Dolo 650'));
    expect(await screen.findByText(/No reliable live source match found/i)).toBeTruthy();
  });

  it('requires a non-empty clarification and resubmits the updated query', async () => {
    searchMedicines
      .mockResolvedValueOnce({ status: 'clarification_required', question: 'Add a brand or form detail.' })
      .mockResolvedValueOnce({ status: 'matched', match_source: 'tinyfish', medicine: { name: 'Metformin' } });
    const user = userEvent.setup();
    render(<SearchPage />);

    await user.type(screen.getByRole('searchbox', { name: /search medicine/i }), 'metformin 500');
    await user.click(screen.getByRole('button', { name: 'Search', exact: true }));
    const detail = await screen.findByRole('textbox', { name: /add a brand or dosage-form detail/i });
    const clarificationForm = detail.closest('form');
    const clarifyButton = within(clarificationForm).getByRole('button', { name: 'Search', exact: true });

    expect(clarifyButton.disabled).toBe(true);
    fireEvent.change(detail, { target: { value: '   ' } });
    expect(clarifyButton.disabled).toBe(true);
    expect(searchMedicines).toHaveBeenCalledTimes(1);

    await user.clear(detail);
    await user.type(detail, 'Glycomet');
    await user.click(clarifyButton);
    await waitFor(() => expect(searchMedicines).toHaveBeenLastCalledWith('metformin 500 Glycomet'));
  });

  it('shows backend errors in the existing error state', async () => {
    searchMedicines.mockResolvedValue({ status: 'error', message: 'Live retrieval is temporarily unavailable.' });
    const user = userEvent.setup();
    render(<SearchPage />);

    await user.type(screen.getByRole('searchbox', { name: /search medicine/i }), 'Dolo 650');
    await user.click(screen.getByRole('button', { name: 'Search', exact: true }));

    expect(await screen.findByText('Live retrieval is temporarily unavailable.')).toBeTruthy();
  });
});
