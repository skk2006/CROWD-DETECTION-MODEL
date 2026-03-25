document.addEventListener('DOMContentLoaded', () => {
    const uploadArea = document.getElementById('uploadArea');
    const imageInput = document.getElementById('imageInput');
    const imagePreview = document.getElementById('imagePreview');
    const uploadPlaceholder = document.getElementById('uploadPlaceholder');
    const openCameraBtn = document.getElementById('openCameraBtn');
    const captureBtn = document.getElementById('captureBtn');
    const closeCameraBtn = document.getElementById('closeCameraBtn');
    const cameraFeed = document.getElementById('cameraFeed');
    const cameraCanvas = document.getElementById('cameraCanvas');
    const uploadForm = document.getElementById('uploadForm');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const submitBtn = document.getElementById('submitBtn');
    const successMessage = document.getElementById('successMessage');
    const placeInput = document.getElementById('placeInput');
    const placeList = document.getElementById('placeList');
    const comboboxToggle = document.getElementById('comboboxToggle');
    const countBtn = document.getElementById('countBtn');
    const headCountResult = document.getElementById('headCountResult');

    let stream = null;
    let lastHeadCount = null;

    // Count button behavior (show latest head count)
    countBtn.addEventListener('click', () => {
        if (lastHeadCount === null) {
            ipsAlert('Please upload an image first then retry.', 'info');
            return;
        }
        headCountResult.textContent = 'Head count: ' + lastHeadCount;
        ipsAlert('Detected head count: ' + lastHeadCount, 'success');
    });

    // Combobox functionality
    if (placeInput && placeList && comboboxToggle) {
        const listItems = placeList.querySelectorAll('li');
        
        // Toggle dropdown
        comboboxToggle.addEventListener('click', (e) => {
            e.preventDefault();
            placeList.classList.toggle('show');
            if (placeList.classList.contains('show')) {
                placeInput.focus();
            }
        });
        
        // Show dropdown on input focus
        placeInput.addEventListener('focus', () => {
            placeList.classList.add('show');
            filterList('');
        });
        
        // Filter list as user types
        placeInput.addEventListener('input', () => {
            filterList(placeInput.value);
        });
        
        // Select item on click
        listItems.forEach(item => {
            item.addEventListener('click', () => {
                placeInput.value = item.dataset.value;
                placeList.classList.remove('show');
            });
        });
        
        // Close dropdown when clicking outside
        document.addEventListener('click', (e) => {
            if (!e.target.closest('.combobox-wrapper')) {
                placeList.classList.remove('show');
            }
        });
        
        function filterList(query) {
            const q = query.toLowerCase();
            listItems.forEach(item => {
                const text = item.textContent.toLowerCase();
                if (text.includes(q) || q === '') {
                    item.classList.remove('hidden');
                } else {
                    item.classList.add('hidden');
                }
            });
        }
    }

    // Click to browse files
    uploadArea.addEventListener('click', () => {
        imageInput.click();
    });

    // Drag & drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#2563eb';
    });

    uploadArea.addEventListener('dragleave', () => {
        uploadArea.style.borderColor = '#d1d5db';
    });

    uploadArea.addEventListener('drop', async (e) => {
        e.preventDefault();
        uploadArea.style.borderColor = '#d1d5db';
        if (e.dataTransfer.files.length > 0) {
            try {
                let file = e.dataTransfer.files[0];
                file = await convertHeicIfNeeded(file);
                setFileInput(file);
                showPreview(file);
            } catch (_) {
                // conversion error already shown to user
            }
        }
    });

    // File selected
    imageInput.addEventListener('change', async () => {
        if (imageInput.files.length > 0) {
            try {
                let file = imageInput.files[0];
                file = await convertHeicIfNeeded(file);
                setFileInput(file);
                showPreview(file);
            } catch (_) {
                // conversion error already shown to user
                imageInput.value = '';
            }
        }
    });

    // Convert HEIC/HEIF to JPEG in the browser before uploading
    async function convertHeicIfNeeded(file) {
        const name = (file.name || '').toLowerCase();
        const type = (file.type || '').toLowerCase();
        const isHeic = name.endsWith('.heic') || name.endsWith('.heif') ||
                       type === 'image/heic' || type === 'image/heif' ||
                       type === '';

        // If type is empty, check magic bytes in the file header
        let confirmedHeic = isHeic;
        if (type === '' && !name.endsWith('.heic') && !name.endsWith('.heif')) {
            try {
                const header = await file.slice(0, 12).arrayBuffer();
                const view = new Uint8Array(header);
                // HEIF magic: bytes 4-8 = 'ftyp'
                if (view.length >= 12 &&
                    view[4] === 0x66 && view[5] === 0x74 && view[6] === 0x79 && view[7] === 0x70) {
                    const brand = String.fromCharCode(view[8], view[9], view[10], view[11]).toLowerCase();
                    confirmedHeic = ['heic', 'heix', 'mif1', 'msf1', 'hevc'].includes(brand);
                } else {
                    confirmedHeic = false;
                }
            } catch (_) {
                confirmedHeic = false;
            }
        }

        if (!confirmedHeic || typeof heic2any === 'undefined') return file;

        try {
            const blob = await heic2any({ blob: file, toType: 'image/jpeg', quality: 0.92 });
            const converted = Array.isArray(blob) ? blob[0] : blob;
            const newName = name.replace(/\.heic$|\.heif$/i, '') + '.jpg';
            return new File([converted], newName || 'converted.jpg', { type: 'image/jpeg' });
        } catch (e) {
            console.error('Client-side HEIC conversion failed:', e);
            // Don't silently return original — tell the user
            ipsAlert('Could not convert HEIC image in browser. Please try converting to JPEG first.', 'error');
            throw e;
        }
    }

    // Replace the file in the input element
    function setFileInput(file) {
        const dt = new DataTransfer();
        dt.items.add(file);
        imageInput.files = dt.files;
    }

    function showPreview(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imagePreview.style.display = 'block';
            uploadPlaceholder.style.display = 'none';
        };
        reader.readAsDataURL(file);
    }

    function resetForm() {
        imagePreview.style.display = 'none';
        uploadPlaceholder.style.display = 'block';
        imageInput.value = '';
        uploadForm.reset();
    }

    // Open camera
    openCameraBtn.addEventListener('click', async () => {
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: 'environment', width: 1280, height: 720 }
            });
            cameraFeed.srcObject = stream;
            cameraFeed.style.display = 'block';
            captureBtn.style.display = 'inline-block';
            closeCameraBtn.style.display = 'inline-block';
            openCameraBtn.style.display = 'none';
        } catch (err) {
            ipsAlert('Could not access camera: ' + err.message, 'error');
        }
    });

    // Capture photo
    captureBtn.addEventListener('click', () => {
        cameraCanvas.width = cameraFeed.videoWidth;
        cameraCanvas.height = cameraFeed.videoHeight;
        cameraCanvas.getContext('2d').drawImage(cameraFeed, 0, 0);

        cameraCanvas.toBlob((blob) => {
            const file = new File([blob], 'camera_capture.jpg', { type: 'image/jpeg' });
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);
            imageInput.files = dataTransfer.files;
            showPreview(file);
            closeCamera();
        }, 'image/jpeg', 0.95);
    });

    // Close camera
    closeCameraBtn.addEventListener('click', closeCamera);

    function closeCamera() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
        }
        cameraFeed.style.display = 'none';
        captureBtn.style.display = 'none';
        closeCameraBtn.style.display = 'none';
        openCameraBtn.style.display = 'inline-block';
    }

    // AJAX form submission
    uploadForm.addEventListener('submit', async (e) => {
        e.preventDefault();

        // Validate place
        const place = placeInput ? placeInput.value.trim() : '';
        if (!place) {
            ipsAlert('Please enter or select a place.', 'error');
            return;
        }

        if (!imageInput.files || imageInput.files.length === 0) {
            ipsAlert('Please select or capture an image first.', 'error');
            return;
        }

        loadingOverlay.style.display = 'flex';
        submitBtn.disabled = true;

        // Safety net: convert HEIC at submit time in case it wasn't converted on file select
        try {
            let file = imageInput.files[0];
            file = await convertHeicIfNeeded(file);
            setFileInput(file);
        } catch (_) {
            loadingOverlay.style.display = 'none';
            submitBtn.disabled = false;
            return;
        }

        const formData = new FormData(uploadForm);

        try {
            const response = await fetch(uploadForm.action, {
                method: 'POST',
                body: formData
            });

            let data;
            const contentType = response.headers.get('content-type') || '';
            if (contentType.includes('application/json')) {
                data = await response.json();
            } else {
                const text = await response.text();
                throw new Error(text || 'Server returned an unexpected response.');
            }

            loadingOverlay.style.display = 'none';
            submitBtn.disabled = false;

            if (data.success) {
                // Save head count for manual display via Count button
                lastHeadCount = data.head_count;
                countBtn.disabled = false;
                headCountResult.textContent = 'Head count: ' + lastHeadCount;

                // Show success toast
                successMessage.style.display = 'flex';
                resetForm();
                setTimeout(() => {
                    successMessage.style.display = 'none';
                }, 2000);
            } else {
                ipsAlert(data.error || 'An error occurred.', 'error');
            }
        } catch (err) {
            loadingOverlay.style.display = 'none';
            submitBtn.disabled = false;
            ipsAlert('Upload failed: ' + err.message, 'error');
        }
    });
});
