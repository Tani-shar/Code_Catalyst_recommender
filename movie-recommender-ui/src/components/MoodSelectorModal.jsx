import { Dialog, Transition } from '@headlessui/react';
import { Fragment } from 'react';

export default function MoodSelectorModal({ isOpen, onClose, setMood }) {
  const moodOptions = [
    { value: 'happy', label: 'Happy', emoji: '😊' },
    { value: 'sad', label: 'Sad', emoji: '😢' },
    { value: 'romantic', label: 'Romantic', emoji: '🥰' },
    { value: 'thrilled', label: 'Thrilled', emoji: '🎉' },
    { value: 'adventurous', label: 'Adventurous', emoji: '🌍' },
    { value: 'dark', label: 'Dark', emoji: '🖤' },
    { value: 'curious', label: 'Curious', emoji: '🔍' },
    { value: 'motivated', label: 'Motivated', emoji: '💪' },
  ];

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
                <Dialog.Title className="text-2xl font-bold mb-4">Select Your Mood</Dialog.Title>
                <div className="grid grid-cols-2 gap-3">
                  {moodOptions.map((mood) => (
                    <button
                      key={mood.value}
                      onClick={() => {
                        setMood(mood.value);
                        onClose();
                      }}
                      className="flex flex-col items-center justify-center p-4 bg-gray-800 rounded-xl hover:bg-red-600 transition-all"
                    >
                      <span className="text-3xl mb-2">{mood.emoji}</span>
                      <span className="text-sm font-medium">{mood.label}</span>
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