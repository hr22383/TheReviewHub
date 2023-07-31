<!DOCTYPE html>
<html>
<head>
    <title>Add Product</title>
</head>
<style>
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

    img {
        width: 100px;
        height: 100px;
    }
</style>
<body>

<h2>Add Product</h2>

<form method="POST" action="<?php echo htmlspecialchars($_SERVER["PHP_SELF"]); ?>" enctype="multipart/form-data">
    <label for="product_name">Product Name:</label>
    <input type="text" name="product_name" id="product_name" required><br><br>

    <label for="product_description">Product Description:</label><br>
    <textarea name="product_description" id="product_description" rows="4" cols="50" required></textarea><br><br>

    <label for="product_price">Product Price:</label>
    <input type="number" name="product_price" id="product_price" step="0.01" required><br><br>

    <label for="product_photo">Product Photo:</label>
    <input type="file" name="product_photo" id="product_photo" required><br><br>

    <input type="submit" value="Add Product">
</form>

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
        echo "Product data inserted successfully.";
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
    echo "<tr><th>Product Name</th><th>Product Description</th><th>Product Price</th><th>Product Photo</th></tr>";
    while ($row = mysqli_fetch_assoc($productResult)) {
        echo "<tr>";
        echo "<td>" . $row['product_name'] . "</td>";
        echo "<td><p>" . $row['product_description'] . "</p></td>";
        echo "<td>$" . $row['product_price'] . "</td>";
        echo "<td><img src='data:image/jpeg;base64," . base64_encode($row['product_photo']) . "' alt='Product Photo'></td>";
        echo "</tr>";
    }
    echo "</table>";
} else {
    echo "No products found.";
}
// Close the database connection
mysqli_close($conn);
?>
