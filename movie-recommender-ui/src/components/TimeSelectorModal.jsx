import { Dialog, Transition } from '@headlessui/react';
import { Fragment } from 'react';

export default function TimeSelectorModal({ isOpen, onClose, setTime }) {
  const times = ['morning', 'afternoon', 'evening', 'night'];

  return (
    <Transition appear show={isOpen} as={Fragment}>
      <Dialog as="div" className="relative z-50" onClose={onClose}>
        <Transition.Child
          as={Fragment}
          enter="ease-out duration-300"
          enterFrom="opacity-0"
          enterTo="opacity-100"
          leave="ease-in duration-200"
          leaveFrom="opacity-100"
          leaveTo="opacity-0"
        >
          <div className="fixed inset-0 bg-black bg-opacity-50" />
        </Transition.Child>

        <div className="fixed inset-0 overflow-y-auto">
          <div className="flex min-h-full items-center justify-center p-4">
            <Transition.Child
              as={Fragment}
              enter="ease-out duration-300"
              enterFrom="opacity-0 scale-95"
              enterTo="opacity-100 scale-100"
              leave="ease-in duration-200"
              leaveFrom="opacity-100 scale-100"
              leaveTo="opacity-0 scale-95"
            >
              <Dialog.Panel className="w-full max-w-md bg-gray-900 rounded-2xl p-6 text-white">
                <Dialog.Title className="text-2xl font-bold mb-4">Select Time of Day</Dialog.Title>
                <div className="flex flex-wrap gap-2">
                  {times.map((time) => (
                    <button
                      key={time}
                      onClick={() => {
                        setTime(time);
                        onClose();
                      }}
                      className="px-4 py-2 bg-gray-800 rounded hover:bg-red-600"
                    >
                      {time.charAt(0).toUpperCase() + time.slice(1)}
                    </button>
                  ))}
                </div>
              </Dialog.Panel>
            </Transition.Child>
          </div>
        </div>
      </Dialog>
    </Transition>
  );
}