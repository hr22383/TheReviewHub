<!DOCTYPE html>
<html>
<head>
    <title>Product Review</title>
    <style>
        /* CSS styling code remains the same */

        /* Add a class for the popup container */
        .popup-container {
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(0, 0, 0, 0.5);
            z-index: 9999;
        }
        .popup-content {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background-color: #fff;
            padding: 20px;
            max-width: 600px;
            width: 100%;
            border: 1px solid #ccc;
        }
    </style>
</head>
<body>
    <!-- Add a button to open the popup -->
    <button onclick="openPopup()">Open Review Popup</button>

    <!-- Add the popup container and content -->
    <div class="popup-container" id="popupContainer">
        <div class="popup-content">
            <h1>Product Review</h1>

            <div class="review-form">
                <!-- Review form code remains the same -->
            </div>

            <div class="review-list">
                <!-- Review list code remains the same -->
            </div>
        </div>
    </div>

    <script>
        // JavaScript function to open the popup
        function openPopup() {
            var popup = document.getElementById('popupContainer');
            popup.style.display = 'block';
        }
    </script>
</body>
</html>
