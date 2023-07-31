 <?php
	session_start();
?>

<?php
// Assuming you have a database connection established
// Database configuration
// $host = "localhost"; // Replace with your database host
// $username = "root"; // Replace with your database username
// $password = ""; // Replace with your database password
// $database = "reviewhub"; // Replace with your database name

include 'liveconfig.php';
//include 'localconfig.php';




// Create a database connection
$conn = mysqli_connect($host, $username, $password, $database);

// Check if the connection was successful
if (!$conn) {
    die("Connection failed: " . mysqli_connect_error());
}

// Check if the form is submitted
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    // Check if the session is set (not null)
    if (isset($_SESSION['first_name'])) {
        // Retrieve the data from the form
        $productID = $_POST["product_id"];
        $rating = $_POST["rating"];
        $comment = $_POST["comment"];
        $user_name = $_SESSION['first_name'];

        // Insert the data into the reviews table
		$date = date('Y-m-d'); // Get the current date

$insertSql = "INSERT INTO reviews (product_id, rating, comment, user_name, review_date) VALUES ('$productID', '$rating', '$comment', '$user_name', '$date')";

     //   $insertSql = "INSERT INTO reviews (product_id, rating, comment, user_name) VALUES ('$productID', '$rating', '$comment', '$user_name')";
        if (mysqli_query($conn, $insertSql)) {
            mysqli_close($conn);
            $successMessage = "Review saved successfully.";
            echo "<script>";
            echo "alert('$successMessage');";
            echo "window.location.href = 'index.php';"; // Replace with the URL of your original page
            echo "</script>";
            exit();
        } else {
            echo "Error: " . mysqli_error($conn);
        }
    } else {
        $errorMessage = "You must be logged in to leave a review.";
        echo "<script>";
        echo "alert('$errorMessage');";
        echo "window.location.href = 'login.php';"; // Replace with the URL of your login page
        echo "</script>";
    }
}

mysqli_close($conn);
?>