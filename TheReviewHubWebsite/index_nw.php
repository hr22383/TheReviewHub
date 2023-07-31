<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>Welcome to Review Hub</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="">
    <meta name="author" content="">
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
    </style>
</head>
<body>
<!-- 
Upper Header Section 
-->
<div class="navbar navbar-inverse navbar-fixed-top">
    <div class="topNav">
        <div class="container">
            <div class="alignR">
                <div class="pull-left socialNw">
                    <a href="#"><span class="icon-twitter"></span></a>
                    <a href="#"><span class="icon-facebook"></span></a>
                    <a href="#"><span class="icon-youtube"></span></a>
                    <a href="#"><span class="icon-tumblr"></span></a>
                </div>
            </div>
        </div>
    </div>
</div>

<!--
Lower Header Section 
-->
<div class="container">
    <div id="gototop"> </div>
    <header id="header">
        <div class="row">
            <div class="span4">
                <h1>
                    <a class="logo" href="index.html">
                        <span>Welcome to Review Hub</span>
                        <img src="assets/img/review.jpg" alt="bootstrap sexy shop">
                    </a>
                </h1>
            </div>
        </div>
    </header>

    <!--
    Navigation Bar Section 
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
                        <li class="active"><a href="list-view.html">Home</a></li>
                        <li class=""><a href="about_use.html">About Us</a></li>
                        <li class=""><a href="register_user.html">Register</a></li>
                        <li class=""><a href="login_user.html">Login</a></li>
                    </ul>
                </div>
            </div>
        </div>
    </div>

    <!-- 
    Body Section 
    -->
<div class="search-container">
   
	
	<div id="reviewPopup" style="display: none;">
    <div id="reviewContent">
       
        <h2>Write a Review</h2>
        <form method="POST" action="insert_review.php" class="form-horizontal" enctype="multipart/form-data">

            <input type="hidden" id="product_id" name="product_id">
            <label for="rating">Rating:</label>
            <select name="rating" id="rating">
                <option value="1">1 Star</option>
                <option value="2">2 Stars</option>
                <option value="3">3 Stars</option>
                <option value="4">4 Stars</option>
                <option value="5">5 Stars</option>
            </select>
            <br>
            <label for="comment">Comment:</label>
            <textarea name="comment" id="comment" rows="5"></textarea>
            <br>
            <input type="submit" value="Submit Review">
        </form>
    </div>
</div>

    </form>
</div>

  <?php
// Assuming you have a database connection established
// Database configuration
$host = "localhost"; // Replace with your database host
$username = "root"; // Replace with your database username
$password = ""; // Replace with your database password
$database = "reviewhub"; // Replace with your database name

// Create a database connection
$conn = mysqli_connect($host, $username, $password, $database);

// Check if the connection was successful
if (!$conn) {
    die("Connection failed: " . mysqli_connect_error());
}

// Retrieve product data from the product_test table
$productSql = "SELECT * FROM product_test";
$productResult = mysqli_query($conn, $productSql);

// Display product data in a table
if (mysqli_num_rows($productResult) > 0) {
    echo "<input type='text' class='search-input' placeholder='Search'>";
    echo "<table>";
    echo "<tr><th>Product Name</th><th>Product Description</th><th>Product Price</th><th>Product Photo</th><th>Add Review</th><th>View Review</th></tr>";

    while ($row = mysqli_fetch_assoc($productResult)) {
        echo "<tr class='productRow'>";
        echo "<td>" . $row['product_name'] . "</td>";
        echo "<td><p>" . $row['product_description'] . "</p></td>";
        echo "<td>$" . $row['product_price'] . "</td>";
        echo "<td><img class='product-photo' src='data:image/jpeg;base64," . base64_encode($row['product_photo']) . "' alt='Product Photo'></td>";
echo "<td><button data-product-id='" . $row['product_id'] . "' class='exclusive shopBtn review-btn'>Review</button></td>";
        echo "<td>";
        echo "<div class='review-col'>";
        echo "<div class='review-header'>";

        echo "<a href='#' class='review-toggle'>View More</a>";
        echo "</div>";

        // Retrieve reviews for the current product
        $reviewsSql = "SELECT * FROM reviews WHERE product_id = " . $row['product_id'] . " LIMIT 2";
        $reviewsResult = mysqli_query($conn, $reviewsSql);

        if (mysqli_num_rows($reviewsResult) > 0) {
            echo "<div class='reviews'>";
            echo "<ul class='comment-list'>";

            while ($review = mysqli_fetch_assoc($reviewsResult)) {
                echo "<li class='comment'>" . $review['comment'] . "</li>";
            }

            echo "</ul>";

            // Check if there are more than two reviews
            $reviewCountSql = "SELECT COUNT(*) AS review_count FROM reviews WHERE product_id = " . $row['product_id'];
            $reviewCountResult = mysqli_query($conn, $reviewCountSql);
            $reviewCount = mysqli_fetch_assoc($reviewCountResult)['review_count'];

            if ($reviewCount > 2) {
                echo "<p class='more-reviews'><a href='#'>View All Reviews ($reviewCount)</a></p>";
            }

            echo "</div>";
        } else {
            echo "<div class='reviews'>";
            echo "<p class='no-reviews'>No reviews yet.</p>";
            echo "</div>";
        }

        echo "</div>";
        echo "</td>";
        echo "</tr>";
    }

    echo "</table>";
} else {
    echo "No products found.";
}

mysqli_close($conn);
?>


<?php
// Assuming you have a database connection established
// Database configuration
$host = "localhost"; // Replace with your database host
$username = "root"; // Replace with your database username
$password = ""; // Replace with your database password
$database = "reviewhub"; // Replace with your database name

// Create a database connection
$conn = mysqli_connect($host, $username, $password, $database);

// Check if the connection was successful
if (!$conn) {
    die("Connection failed: " . mysqli_connect_error());
}

// Check if the form is submitted
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    // Retrieve the data from the form
    $productID = $_POST["product_id"];
    $rating = $_POST["rating"];
    $comment = $_POST["comment"];

    // Insert the data into the reviews table
    $insertSql = "INSERT INTO reviews (product_id, rating, comment) VALUES ('$productID', '$rating', '$comment')";
    if (mysqli_query($conn, $insertSql)) {
        echo "Review saved successfully.";
    } else {
        echo "Error: " . mysqli_error($conn);
    }
}

mysqli_close($conn);
?>




</div>

<!-- JavaScript code here -->


<!-- Include jQuery library -->
<script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
<script>
    $(document).ready(function() {
        $(".search-input").on("keyup", function() {
            var value = $(this).val().toLowerCase();
            $(".productRow").filter(function() {
                $(this).toggle($(this).text().toLowerCase().indexOf(value) > -1)
            });
        });
    });
</script>
<script src="https://code.jquery.com/ui/1.12.1/jquery-ui.js"></script>
<!-- Add this JavaScript code to the existing script tag -->
<!-- Add this JavaScript code to the existing script tag -->
<!-- Add this JavaScript code to the existing script tag -->
<!-- Add this JavaScript code to the existing script tag -->
<!-- Add this JavaScript code to the existing script tag -->
<script>
  $(document).ready(function() {
    // Create the review form dialog
    $("#reviewPopup").dialog({
      autoOpen: false,
      modal: true,
      width: 400,
      height: 400,
      resizable: false,
      draggable: false,
      dialogClass: "review-dialog",
      position: { my: "center", at: "top+300", of: window },
    });

    // Open review form popup
    function openReviewForm(productID) {
      $("#product_id").val(productID); // Set the product ID in the hidden field
      $("#reviewPopup").dialog("open"); // Open the dialog
    }

    // Filter product rows based on search input
    $(".search-input").on("keyup", function() {
      var value = $(this).val().toLowerCase();
      $(".productRow").filter(function() {
        $(this).toggle($(this).text().toLowerCase().indexOf(value) > -1);
      });
    });

    // Handle click event on Review button
    $(".review-btn").click(function() {
      var productID = $(this).data("product-id");
      openReviewForm(productID);
    });
  });
</script>


