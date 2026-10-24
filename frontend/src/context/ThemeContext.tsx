import React, { createContext, useContext, useState, useEffect } from 'react';

export type Theme = 'dark' | 'light';

interface ThemeContextType {
  theme: Theme;
  toggleTheme: () => void;
  setTheme: (theme: Theme) => void;
}

const ThemeContext = createContext<ThemeContextType>({
  theme: 'dark',
  toggleTheme: () => {},
  setTheme: () => {},
});

export const ThemeProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [theme, setThemeState] = useState<Theme>(() => {
    try {
      const saved = localStorage.getItem('lung-cancer-ai-theme');
      if (saved === 'light' || saved === 'dark') return saved;
    } catch {}
    return 'dark';
  });

  useEffect(() => {
    try {
      localStorage.setItem('lung-cancer-ai-theme', theme);
    } catch {}

    try {
      document.documentElement.setAttribute('data-theme', theme);
      if (document.body) {
        if (theme === 'light') {
          document.body.classList.remove('bg-slate-950', 'text-slate-100');
          document.body.classList.add('bg-slate-50', 'text-slate-900', 'theme-light');
        } else {
          document.body.classList.remove('bg-slate-50', 'text-slate-900', 'theme-light');
          document.body.classList.add('bg-slate-950', 'text-slate-100');
        }
      }
    } catch {}
  }, [theme]);

  const toggleTheme = () => {
    setThemeState(prev => (prev === 'dark' ? 'light' : 'dark'));
  };

  const setTheme = (t: Theme) => setThemeState(t);

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme, setTheme }}>
      {children}
    </ThemeContext.Provider>
  );
};

export const useTheme = () => useContext(ThemeContext);
