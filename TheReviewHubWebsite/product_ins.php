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
	
	<script>
        // JavaScript code here
        function showPopup() {
            alert("Product data inserted successfully.");
        }
    </script>
	
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
</style>
  </head>
<body>
<?php
include 'header.php';
?>

<!-- 
Body Section 
-->

	<form method="POST" class="form-horizontal" action="<?php echo htmlspecialchars($_SERVER["PHP_SELF"]); ?>" enctype="multipart/form-data">


		<div class="control-group">
			<label class="control-label" for="inputFname">Product Name: <sup>*</sup></label>
			<div class="controls">
			  <input type="text" name="product_name" id="product_name" required>
			</div>
		 </div>
		 	 <div class="control-group">
			<label class="control-label" for="inputFname">Product Price: <sup>*</sup></label>
			<div class="controls">
			   <input type="number" name="product_price" id="product_price"  required>
			</div>
		 </div>
		<div class="control-group">
			<label class="control-label" for="inputFname">Product Description: <sup>*</sup></label>
			<div class="controls">
			  <textarea name="product_description" id="product_description" rows="3" cols="150" required></textarea>
			</div>
		 </div>
	
		  <div class="control-group">
			<label class="control-label" for="inputFname">Product Photo: <sup>*</sup></label>
			<div class="controls">
			  <input type="file" name="product_photo" id="product_photo" required><br><br>
			</div>
			<div class="controls">
		
		   <input type="submit" value="Add Product"  class="exclusive shopBtn" name="submitAccount">
		</div>
		 </div>
   
<div class="control-group">
		
	</div>
   
    


    

  
		
		
		
		
<?php
// Assuming you have a database connection established
// Database configuration

// $host = "sql100.infinityfree.com"; // Replace with your database host
// $username = "if0_34355172"; // Replace with your database username
// $password = "gbcIkmvbff"; // Replace with your database password
// $database = "if0_34355172_reviewhub"; // Replace with your database name

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
    // Retrieve and sanitize form data
    $productName = mysqli_real_escape_string($conn, $_POST["product_name"]);
    $productDescription = mysqli_real_escape_string($conn, $_POST["product_description"]);
    $productPrice = mysqli_real_escape_string($conn, $_POST["product_price"]);

    // Check if a file is uploaded
    if (isset($_FILES["product_photo"]) && $_FILES["product_photo"]["error"] == UPLOAD_ERR_OK) {
        $photoData = mysqli_real_escape_string($conn, file_get_contents($_FILES["product_photo"]["tmp_name"]));
    }

    // Insert data into the product_test table
    $productSql = "INSERT INTO product_test (product_name, product_description, product_price, product_photo)
                   VALUES ('$productName', '$productDescription', '$productPrice', '$photoData')";
    $productResult = mysqli_query($conn, $productSql);

    if ($productResult) {
        echo "<script>showPopup();</script>"; 
    } else {
        echo "Error inserting product data: " . mysqli_error($conn);
    }
}

// Retrieve product data from the product_test table
$productSql = "SELECT * FROM product_test";
$productResult = mysqli_query($conn, $productSql);

// Display product data in a table
if (mysqli_num_rows($productResult) > 0) {
    echo "<table>";
    echo "<tr><th>Product Name</th><th>Product Description</th><th>Product Price</th><th>Product Photo</th><th>Delete</th></tr>";
    while ($row = mysqli_fetch_assoc($productResult)) {
       echo "<tr>";
        echo "<td>" . $row['product_name'] . "</td>";
        echo "<td><p>" . $row['product_description'] . "</p></td>";
        echo "<td>$" . $row['product_price'] . "</td>";
        echo "<td><img class='product-photo' src='data:image/jpeg;base64," . base64_encode($row['product_photo']) . "' alt='Product Photo'></td>";
        echo "<td><button onclick='deleteProduct(" . $row['product_id'] . ")' class='exclusive shopBtn'>Delete</button></td>";
        echo "</tr>";
    }
    echo "</table>";
} else {
    echo "No products found.";
}
// Close the database connection
mysqli_close($conn);
?>

		
		</div>
</form>

