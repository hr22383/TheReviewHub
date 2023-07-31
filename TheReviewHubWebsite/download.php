
<?php
include 'liveconfig.php';
//include 'localconfig.php';// Include the file with database connection details

// Function to sanitize the input for CSV export
function sanitizeForCSV($data)
{
    // Remove any double quotes
    $sanitizedData = str_replace('"', '', $data);
    return $sanitizedData;
}


if (isset($_POST['submit'])) {
    $startDate = $_POST['start_date'];
    $endDate = $_POST['end_date'];

    // Retrieve reviews from the database based on the date range
    $reviewsSql = "SELECT comment FROM reviews 
                   JOIN product_test ON reviews.product_id = product_test.product_id
                   WHERE reviews.review_date >= '$startDate' AND reviews.review_date <= '$endDate'";
    $reviewsResult = mysqli_query($conn, $reviewsSql);

    // Check if there are any reviews
    if (mysqli_num_rows($reviewsResult) > 0) {
        // Create a temporary file for storing the CSV data
        $tempFile = tempnam(sys_get_temp_dir(), 'reviews');
        $fileHandle = fopen($tempFile, 'w');

        $fileHandle = fopen($tempFile, 'w');

// Write the CSV headers directly to the file
$headers = array('Comment');
fputcsv($fileHandle, $headers);

// Rest of the code...

		
		
		
        // Write the review data to the CSV file
        while ($review = mysqli_fetch_assoc($reviewsResult)) {
            $sanitizedComment = sanitizeForCSV($review['comment']);

            // Write the review data to the CSV file
            $rowData = array($sanitizedComment);
            fputcsv($fileHandle, $rowData);
        }

        fclose($fileHandle);

        // Set the appropriate headers for CSV file download
        header('Content-Type: text/csv');
        header('Content-Disposition: attachment; filename="reviews.csv"');
        header('Pragma: no-cache');
        header('Expires: 0');
        readfile($tempFile);

        // Delete the temporary file
        unlink($tempFile);
        exit(); // Terminate the script after file download
    } else {
        // No reviews found
        echo "<div class='dashboard-item'>";
        echo "<h3>No reviews found for the selected date range.</h3>";
        echo "</div>";
    }
}

mysqli_close($conn);
?>
<!DOCTYPE html>
<html lang="en">
<head>
   <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
	 <title>Welcome to Review Hub</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
	  <meta charset="utf-8">
    <meta name="description" content="">
    <meta name="author" content="">
    <title>Review Hub</title>
    <style>
        /* Add CSS styles for the header and navigation bar */
        /* ... */

        /* Add CSS styles for the body section */
        /* ... */

        /* Add CSS styles for the review form popup */
        /* ... */

        /* Add CSS styles for responsiveness */
        @media only screen and (max-width: 768px) {
            /* Adjust styles for small screens */
            /* ... */
        }

        @media only screen and (max-width: 480px) {
            /* Adjust styles for extra small screens */
            /* ... */
        }
    </style>


  
   
    <!-- Bootstrap styles -->
    <link href="assets/css/bootstrap.css" rel="stylesheet"/>
    <!-- Customize styles -->
    <link href="style.css" rel="stylesheet"/>
    <!-- font awesome styles -->
    <link href="assets/font-awesome/css/font-awesome.css" rel="stylesheet">
    <!--[if IE 7]>
        <link href="css/font-awesome-ie7.min.css" rel="stylesheet">
    <![endif]-->

    <!--[if lt IE 9]>
        <script src="http://html5shim.googlecode.com/svn/trunk/html5.js"></script>
    <![endif]-->

    <!-- Favicons -->
    <link rel="shortcut icon" href="assets/ico/favicon.ico">

    <style>
	
	
	
		.review-dialog {
    border: 2px dashed #ccc;
    background-color: #f9f9f9;
    padding: 20px;
    color: #333;
  }

  .review-dialog h2 {
    color: #ff0000;
    margin-top: 0;
    padding-bottom: 10px;
    border-bottom: 1px solid #ccc;
  }

  .review-dialog input[type="text"],
  .review-dialog textarea {
    border: 1px solid #ccc;
    padding: 5px;
    width: 100%;
    margin-bottom: 10px;
  }

  .review-dialog input[type="submit"] {
    background-color: #007bff;
    color: #fff;
    border: none;
    padding: 8px 20px;
    cursor: pointer;
  }

  .review-dialog input[type="submit"]:hover {
    background-color: #0056b3;
  }
        table {
            width: 100%;
            border-collapse: collapse;
            border: 1px solid black; /* Set table border color and width */
        }

        th, td {
            padding: 8px;
            text-align: left;
            border-bottom: 1px solid #ddd;
            vertical-align: top;
            border: 1px solid black; /* Set cell border color and width */
        }

        td p {
            white-space: normal;
            overflow-wrap: break-word;
            word-wrap: break-word;
            -ms-word-break: break-all;
            word-break: break-all;
            word-break: break-word;
            line-height: 1; /* Remove extra space */
        }

        .product-photo {
            width: 100px;
            height: 100px;
        }

        .search-container {
            margin-bottom: 20px;
        }

        .search-container input[type="text"] {
            width: 200px;
            padding: 5px;
        }

        .search-container input[type="submit"] {
            padding: 5px 10px;
            background-color: #4CAF50;
            color: white;
            border: none;
            cursor: pointer;
        }
		.logo {
  display: flex;
  flex-direction: column;
  align-items: center;
  text-align: center;
}
    </style>
	
	
</head>
<body>


 
	<h1 class="logo">
  <a href="index.php">
    <span>Welcome to Review Hub</span>
    <img src="assets/img/review.jpg" alt="bootstrap sexy shop" >
  </a>
</h1>


<div class="container">
   
    

    <!--
    Navigation Bar Section 
	 <li class=""><a href="Updatepassword.html">Manage Password</a></li>
	 <li class=""><a href="add_BusinessService.html">Business  Service</a></li>
	<li class=""><a href="add_ProductCategory.html">Product Category</a></li>
	 
    -->
    <div class="navbar">
        <div class="navbar-inner">
            <div class="container">
                <a data-target=".nav-collapse" data-toggle="collapse" class="btn btn-navbar">
                    <span class="icon-bar"></span>
                    <span class="icon-bar"></span>
                    <span class="icon-bar"></span>
                </a>
                <div class="nav-collapse">
             <ul class="nav">
			  <li class="active"><a href="dashboard.php">DashBoard</a></li>
			  <li class=""><a href="profile.php">Profile</a></li>
			  <li class=""><a href="product_ins.php">Add Product</a></li>
			  <li class=""><a href="viewreview.php">View Review</a></li>
			    <li class=""><a href="download.php">Download</a></li>
			  <li class=""><a href="index.php">Logout</a></li>
			</ul>
                </div>
            </div>
        </div>
    </div>
	
    <!-- Your navigation bar code -->
    <!-- ... -->
    <div class="container">
        <div class="dashboard-item">
            <h3>Download Reviews CSV</h3>
            <form method="post" action="">
                <label for="start_date">Start Date:</label>
                <input type="date" id="start_date" name="start_date" required>
                <br>
                <label for="end_date">End Date:</label>
                <input type="date" id="end_date" name="end_date" required>
                <br>
                <input type="submit" name="submit" value="Download CSV">
            </form>
        </div>
    </div>
</body>
</html>
