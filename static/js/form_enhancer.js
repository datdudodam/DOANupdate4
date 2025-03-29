/**
 * Form Enhancer
 * Cung cấp các chức năng nâng cao cho biểu mẫu
 */

// Khởi tạo khi trang đã tải xong
document.addEventListener('DOMContentLoaded', function() {
    // Thêm nút tự động điền vào giao diện
    addAutoFillButton();
    
    // Thêm nút gợi ý nâng cao cho mỗi trường
    addEnhancedSuggestionButtons();
});

// Tự động điền biểu mẫu dựa trên lịch sử
function autoFillForm() {
    // Lấy danh sách các trường trong biểu mẫu hiện tại
    const formFields = [];
    document.querySelectorAll('form input[type="text"]').forEach(input => {
        if (input.id && input.id.match(/\[_\d+_\]/)) {
            formFields.push(input.id);
        }
    });
    
    if (formFields.length === 0) {
        console.log('Không tìm thấy trường nào trong biểu mẫu');
        return;
    }
    
    // Hiển thị thông báo đang tải
    const loadingToast = showToast('Đang tự động điền biểu mẫu...', 'info', 0);
    
    // Gọi API để lấy dữ liệu tự động điền
    fetch('/auto_fill_form', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
            target_fields: formFields
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        // Ẩn thông báo đang tải
        hideToast(loadingToast);
        
        if (data.auto_fill_data && Object.keys(data.auto_fill_data).length > 0) {
            // Điền dữ liệu vào biểu mẫu
            let filledCount = 0;
            for (const [fieldCode, value] of Object.entries(data.auto_fill_data)) {
                const input = document.getElementById(fieldCode);
                if (input && !input.value) {
                    input.value = value;
                    highlightField(input);
                    filledCount++;
                }
            }
            
            // Hiển thị thông báo thành công
            if (filledCount > 0) {
                showToast(`Đã tự động điền ${filledCount} trường dựa trên lịch sử`, 'success');
            } else {
                showToast('Không có trường nào được điền tự động', 'info');
            }
        } else {
            showToast('Không có dữ liệu phù hợp để điền tự động', 'info');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        hideToast(loadingToast);
        showToast('Có lỗi xảy ra khi tự động điền biểu mẫu', 'error');
    });
}

// Tải gợi ý nâng cao cho một trường
function loadEnhancedSuggestions(fieldCode) {
    const suggestionsList = document.getElementById(`suggestions-${fieldCode}`);
    const input = document.getElementById(fieldCode);
    const button = document.querySelector(`button[data-enhanced-suggestions="${fieldCode}"]`);
    const loadingIcon = button.querySelector('.enhanced-suggestion-loading');
    const suggestionIcon = button.querySelector('.enhanced-suggestion-icon');
    const errorDiv = document.getElementById(`error-${fieldCode}`);
    
    // Lấy tất cả giá trị hiện tại của form
    const formData = {};
    document.querySelectorAll('form input[type="text"]').forEach(input => {
        if (input.value.trim()) {
            formData[input.id] = input.value.trim();
        }
    });
    
    // Lấy văn bản ngữ cảnh từ các trường đã điền
    let contextText = Object.values(formData).join(' ');

    // Toggle loading state
    loadingIcon.classList.remove('hidden');
    suggestionIcon.classList.add('hidden');
    button.disabled = true;
    errorDiv.classList.add('hidden');

    fetch('/get_enhanced_suggestions', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
            field_code: fieldCode,
            partial_form: formData,
            context_text: contextText
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        suggestionsList.innerHTML = '';
        if (data.suggestions && data.suggestions.length > 0) {
            data.suggestions.forEach((suggestion, index) => {
                const li = document.createElement('li');
                li.className = 'px-4 py-3 hover:bg-indigo-50 cursor-pointer transition-colors duration-200 touch-manipulation';
                li.textContent = suggestion;
                li.setAttribute('role', 'option');
                li.setAttribute('tabindex', '0');
                li.onclick = () => {
                    input.value = suggestion;
                    suggestionsList.classList.add('hidden');
                    input.focus();
                    highlightField(input);
                };
                li.onkeypress = (e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                        input.value = suggestion;
                        suggestionsList.classList.add('hidden');
                        input.focus();
                        highlightField(input);
                    }
                };
                suggestionsList.appendChild(li);
            });
            suggestionsList.classList.remove('hidden');
            suggestionsList.setAttribute('role', 'listbox');
            suggestionsList.setAttribute('aria-label', 'Danh sách gợi ý nâng cao');
        } else {
            // Hiển thị thông báo lỗi chi tiết nếu có
            if (data.error_details) {
                errorDiv.textContent = `Không có gợi ý nâng cao: ${data.error_details}`;
            } else {
                errorDiv.textContent = 'Không có gợi ý nâng cao nào cho trường này';
            }
            errorDiv.classList.remove('hidden');
        }
    })
    .catch(error => {
        console.error('Error:', error);
        errorDiv.textContent = 'Có lỗi xảy ra khi tải gợi ý nâng cao. Vui lòng thử lại.';
        errorDiv.classList.remove('hidden');
    })
    .finally(() => {
        loadingIcon.classList.add('hidden');
        suggestionIcon.classList.remove('hidden');
        button.disabled = false;
    });
}

// Hiển thị thông báo toast
function showToast(message, type = 'info', duration = 3000) {
    const toast = document.createElement('div');
    toast.className = `fixed bottom-4 right-4 px-6 py-3 rounded-lg shadow-lg z-50 transition-all duration-300 transform translate-y-0 opacity-100`;
    
    // Thiết lập màu sắc dựa trên loại thông báo
    switch (type) {
        case 'success':
            toast.classList.add('bg-green-500', 'text-white');
            break;
        case 'error':
            toast.classList.add('bg-red-500', 'text-white');
            break;
        case 'warning':
            toast.classList.add('bg-yellow-500', 'text-white');
            break;
        default: // info
            toast.classList.add('bg-blue-500', 'text-white');
            break;
    }
    
    toast.innerHTML = `
        <div class="flex items-center">
            <span class="mr-2">
                ${type === 'success' ? '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>' : ''}
                ${type === 'error' ? '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"></path></svg>' : ''}
                ${type === 'warning' ? '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path></svg>' : ''}
                ${type === 'info' ? '<svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>' : ''}
            </span>
            <span>${message}</span>
        </div>
    `;
    
    document.body.appendChild(toast);
    
    // Tự động ẩn thông báo sau khoảng thời gian
    if (duration > 0) {
        setTimeout(() => {
            hideToast(toast);
        }, duration);
    }
    
    return toast;
}

// Ẩn thông báo toast
function hideToast(toast) {
    if (!toast) return;
    
    toast.classList.add('translate-y-2', 'opacity-0');
    setTimeout(() => {
        if (toast.parentNode) {
            toast.parentNode.removeChild(toast);
        }
    }, 300);
}

// Highlight trường đã được điền tự động
function highlightField(input) {
    input.classList.add('bg-indigo-50', 'border-indigo-300');
    setTimeout(() => {
        input.classList.remove('bg-indigo-50', 'border-indigo-300');
    }, 2000);
}

// Thêm nút tự động điền vào giao diện
function addAutoFillButton() {
    const form = document.querySelector('form');
    if (!form) return;
    
    const buttonsContainer = form.querySelector('div:last-child');
    if (!buttonsContainer) return;
    
    const autoFillButton = document.createElement('button');
    autoFillButton.type = 'button';
    autoFillButton.className = 'w-full bg-gradient-to-r from-indigo-600 to-indigo-500 text-white py-4 rounded-xl hover:from-indigo-700 hover:to-indigo-600 focus:ring-4 focus:ring-indigo-200 transition-all duration-300 flex items-center justify-center space-x-2 font-semibold shadow-lg shadow-indigo-200/50 hover:shadow-indigo-300/50 hover:transform hover:-translate-y-0.5 mt-4';
    autoFillButton.innerHTML = `
        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"></path>
        </svg>
        <span>Tự Động Điền</span>
    `;
    autoFillButton.onclick = autoFillForm;
    
    buttonsContainer.appendChild(autoFillButton);
}

// Thêm nút gợi ý nâng cao cho mỗi trường
function addEnhancedSuggestionButtons() {
    document.querySelectorAll('.form-input-group').forEach(group => {
        const input = group.querySelector('input[type="text"]');
        if (!input || !input.id || !input.id.match(/\[_\d+_\]/)) return;
        
        const fieldCode = input.id;
        const buttonsContainer = group.querySelector('.relative > .relative');
        if (!buttonsContainer) return;
        
        const enhancedButton = document.createElement('button');
        enhancedButton.type = 'button';
        enhancedButton.className = 'suggestion-button px-4 py-2 bg-indigo-50 hover:bg-indigo-100 text-indigo-600 rounded-xl transition-all duration-300 flex items-center justify-center shadow-sm hover:shadow focus:ring-2 focus:ring-indigo-200 touch-manipulation relative group';
        enhancedButton.title = 'Gợi ý nâng cao';
        enhancedButton.setAttribute('aria-label', `Gợi ý nâng cao cho ${fieldCode}`);
        enhancedButton.setAttribute('data-enhanced-suggestions', fieldCode);
        enhancedButton.innerHTML = `
            <span class="enhanced-suggestion-loading hidden">
                <svg class="animate-spin h-5 w-5" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
                    <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
            </span>
            <span class="enhanced-suggestion-icon">
                <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z"></path>
                </svg>
            </span>
            <div class="absolute invisible group-hover:visible bg-indigo-600 text-white text-xs rounded py-1 px-2 -top-8 left-1/2 transform -translate-x-1/2 whitespace-nowrap">
                Gợi ý nâng cao cho ${fieldCode}
            </div>
        `;
        enhancedButton.onclick = () => loadEnhancedSuggestions(fieldCode);
        
        // Thêm nút vào container
        buttonsContainer.appendChild(enhancedButton);
    });
}