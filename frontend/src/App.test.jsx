import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import App from './App.jsx';

afterEach(cleanup);

function renderApp() {
  return render(
    <MemoryRouter>
      <App />
    </MemoryRouter>,
  );
}

describe('mobile navigation controls', () => {
  it('toggles the menu and closes it after choosing a navigation link', () => {
    renderApp();
    const toggle = screen.getByRole('button', { name: 'Open menu' });
    const navigation = screen.getByRole('navigation', { name: 'Main navigation' });

    fireEvent.click(toggle);
    expect(screen.getByRole('button', { name: 'Close menu' }).getAttribute('aria-expanded')).toBe('true');
    expect(navigation.classList.contains('nav-open')).toBe(true);

    fireEvent.click(within(navigation).getByRole('link', { name: 'About' }));
    expect(screen.getByRole('button', { name: 'Open menu' }).getAttribute('aria-expanded')).toBe('false');
    expect(navigation.classList.contains('nav-open')).toBe(false);
  });

  it('closes on Escape and returns focus to the menu toggle', () => {
    renderApp();
    const toggle = screen.getByRole('button', { name: 'Open menu' });
    fireEvent.click(toggle);
    fireEvent.keyDown(document, { key: 'Escape' });

    const closedToggle = screen.getByRole('button', { name: 'Open menu' });
    expect(closedToggle.getAttribute('aria-expanded')).toBe('false');
    expect(document.activeElement).toBe(closedToggle);
  });

  it('closes when the user taps outside the header', () => {
    renderApp();
    fireEvent.click(screen.getByRole('button', { name: 'Open menu' }));
    fireEvent.pointerDown(document.querySelector('.hero'));

    expect(screen.getByRole('button', { name: 'Open menu' }).getAttribute('aria-expanded')).toBe('false');
  });
});
