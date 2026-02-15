import { ReactNode, useState, useEffect } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { useStore } from '@/store/useStore';

interface LayoutProps {
  children: ReactNode;
}

export function Layout({ children }: LayoutProps) {
  const location = useLocation();
  const user = useStore((state) => state.user);
  const [currentTime, setCurrentTime] = useState(new Date());

  // Update time every second
  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  const navigation = [
    { name: 'Dashboard', path: '/', roles: ['operator', 'engineer', 'admin'] },
    { name: 'Calibrant', path: '/calibration', roles: ['operator', 'engineer', 'admin'] },
    { name: 'Measurement', path: '/measurement', roles: ['operator', 'engineer', 'admin'] },
    { name: 'History', path: '/history', roles: ['operator', 'engineer', 'admin'] },
    { name: 'Maintenance', path: '/maintenance', roles: ['engineer', 'admin'] },
    { name: 'Admin', path: '/admin', roles: ['admin'] },
  ];

  const filteredNav = navigation.filter((item) => 
    user && item.roles.includes(user.role.toLowerCase())
  );

  return (
    <div className="h-screen w-screen bg-[#F0F0F0] flex flex-col overflow-hidden">
      {/* Status Bar - Fixed 43px height, full width */}
      <div className="h-[43px] bg-white border-b border-gray-200 flex items-center px-3">
        <div className="flex-shrink-0 flex items-center gap-2">
          <img src="/EosDxLogo.png" alt="EosDx" className="h-7" />
          <span className="text-base font-['Inter'] whitespace-nowrap">{user?.username || '[Technician Name]'}</span>
        </div>
        <div className="flex-1 text-center">
          <span className="text-lg font-['Inter']">Omniscan Medical XRD Diagnostic System</span>
        </div>
        <div className="flex-shrink-0 text-center">
          <span className="text-base font-['Inter'] whitespace-nowrap">
            {currentTime.toLocaleDateString()} {currentTime.toLocaleTimeString()}
          </span>
        </div>
        <div className="flex-shrink-0 text-right ml-4">
          <span className="text-base font-['Inter'] cursor-pointer hover:text-blue-600">[LogOut]</span>
        </div>
      </div>

      {/* Main content area with navigation and content */}
      <div className="flex flex-1 overflow-hidden">
        {/* Left Navigation Panel */}
        <div className="w-[280px] flex-shrink-0 bg-[#ECECEC] h-full overflow-y-auto">
          <nav className="flex flex-col">
            {filteredNav.map((item) => {
              const isActive = location.pathname === item.path;
              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`px-5 py-3 text-left text-base font-medium transition-colors ${
                    isActive
                      ? 'bg-[#D0D0D0] text-gray-900'
                      : 'text-gray-700 hover:bg-[#E0E0E0]'
                  }`}
                >
                  {item.name}
                </Link>
              );
            })}
          </nav>
        </div>

        {/* Main Content Area */}
        <main className="flex-1 bg-white p-2 overflow-y-auto">
          {children}
        </main>
      </div>
    </div>
  );
}
