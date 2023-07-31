

<!DOCTYPE html>
<html lang="en">

<head>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        /* Add your custom styles here */
        .dialog-box {
            display: none;
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background-color: #fff;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
            z-index: 9999;
        }
        .dialog-box .review-list {
            list-style: none;
            padding: 0;
        }
        .dialog-box .review-list li {
            margin-bottom: 10px;
        }
    </style>
	
<style>
.popup-dialog {
  position: fixed;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%);
  width: 500px;
  max-height: 80vh;
  overflow-y: auto;
  background-color: #f1f1f1;
  border-radius: 10px;
  box-shadow: 0 0 10px rgba(0, 0, 0, 0.3);
  z-index: 9999;
  font-family: Arial, sans-serif;
  box-sizing: border-box;
}

.popup-header {
  padding: 20px;
  background-color: #4CAF50;
  border-radius: 10px 10px 0 0;
  color: #fff;
}

.popup-header h2 {
  font-size: 24px;
  margin: 0;
}

.popup-content {
  padding: 20px;
}

.popup-dialog p {
  color: #555;
  line-height: 1.5;
}

.popup-dialog button {
  display: block;
  margin-top: 20px;
  padding: 8px 16px;
  background-color: #4CAF50;
  color: #fff;
  border: none;
  border-radius: 4px;
  font-size: 16px;
  cursor: pointer;
}

.product-photo-container {
  position: relative;
  width: 80px; /* Adjust the width as needed */
  height: 80px; /* Adjust the height as needed */
}

.product-photo {
  width: 100%;
  height: 100%;
  object-fit: cover;
  transition: transform 0.3s ease;
}

.product-photo:hover {
  transform: scale(1.25); /* Adjust the scale factor for the desired enlargement */
}

.product-photo:hover {
  z-index: 1;
  position: absolute;
  top: 50%;
  left: 50%;
  transform: translate(-50%, -50%) scale(2.5); /* Adjust the scale factor for the desired enlargement */
  width: 500px; /* Adjust the width as needed */
  height: 140px; /* Adjust the height as needed */
}

/* Table */
/* Table */
.product-table {
  width: 100%;
  border-collapse: collapse;
  border: 1px solid #ddd; /* Add table border */
}

.product-table th,
.product-table td {
  padding: 10px;
  border: 1px solid #ddd;
  background-color: #f5f5f5; /* Set the desired background color */
}
/* Product Photo */
.product-photo-container {
  width: 80px;
}

.product-photo {
  max-width: 100%;
  height: auto;
}

/* Product Description */
.product-description {
  width: 500px;
}

/* Add Review Button */
.review-btn {
  display: block;
  width: 100%;
  padding: 5px 10px;
  background-color: #f5f5f5;
  border: 1px solid #ddd;
  border-radius: 4px;
  text-align: center;
  cursor: pointer;
}

/* View Review */
.review-col {
  width: 300px;
}

.review-header {
  margin-bottom: 10px;
}

.reviews {
  padding: 10px;
  background-color: #f5f5f5;
  border: 1px solid #ddd;
  overflow: auto; /* Add overflow property for long reviews */
  max-height: 200px; /* Set a maximum height to limit the size */
}

.comment-list {
  list-style-type: none;
  padding: 0;
}

.comment {
  margin-bottom: 0px;
}

.more-reviews {
  margin-top: 0px;
  text-align: right;
}

.view-all-reviews {
  color: blue;
  text-decoration: underline;
  cursor: pointer;
}

.no-reviews {
  color: #777;
}
.rating-bar {
  display: inline-block;
  width: 150px;
  height: 20px;
  background-color: #ccc;
}

.rating-value {
  height: 100%;
  background-color: #ffcc00;
}

.rating-number {
  margin-left: 10px;
}




</style>





   
</head>


<body>
<?php
include 'heade_user.php';
?>
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

include 'liveconfig.php';
//include 'localconfig.php';

// Retrieve product data from the product_test table
$productSql = "SELECT * FROM product_test";
$productResult = mysqli_query($conn, $productSql);

// Display product data in a table
if (mysqli_num_rows($productResult) > 0) {
    echo "<input type='text' class='search-input' placeholder='Search' style='margin-bottom: 10px; margin-right: 120px;'>";

    echo "<table class='product-table'>";
    echo "<tr><th>Product Photo</th><th>Product Description</th><th>Add Review</th><th>View Review</th></tr>";

    while ($row = mysqli_fetch_assoc($productResult)) {
        echo "<tr class='productRow'>";

        // Set a shorter width for the product photo cell
        echo "<td style='width: 80px;'>";
        echo "<div class='product-photo-container'>";
        echo "<img class='product-photo' src='data:image/jpeg;base64," . base64_encode($row['product_photo']) . "' alt='Product Photo'>";
        echo "</div>";
        echo "</td>";

        // Set a longer width for the product description cell
        echo "<td style='width: 500px;'><p>" . $row['product_description'] . "</p></td>";

        // Set a shorter width for the add review cell
        echo "<td style='width: 80px;'>";
        echo "<button data-product-id='" . $row['product_id'] . "' class='exclusive shopBtn review-btn'>Review</button>";
        echo "</td>";

        // Set a longer width for the view review cell
        echo "<td style='width: 300px;'>";
        echo "<div class='review-col'>";
        echo "<div class='review-header'></div>";

        // Retrieve reviews for the current product
        $reviewsSql = "SELECT reviews.*, user_new.first_name FROM reviews JOIN user_new ON reviews.user_name = user_new.first_name WHERE reviews.product_id = " . $row['product_id'];
        $reviewsResult = mysqli_query($conn, $reviewsSql);

        // Retrieve review count for the current product
        $reviewCountSql = "SELECT COUNT(*) AS review_count FROM reviews WHERE product_id = " . $row['product_id'];
        $reviewCountResult = mysqli_query($conn, $reviewCountSql);
        $reviewCount = mysqli_fetch_assoc($reviewCountResult)['review_count'];

        if (mysqli_num_rows($reviewsResult) > 0) {
            echo "<div class='reviews'>";
            echo "<ul class='comment-list'>";

            // Retrieve average rating for the current product
            $averageRatingSql = "SELECT AVG(rating) AS average_rating FROM reviews WHERE product_id = " . $row['product_id'];
            $averageRatingResult = mysqli_query($conn, $averageRatingSql);
            $averageRating = mysqli_fetch_assoc($averageRatingResult)['average_rating'];

            echo "<div class='rating-bar'>";
            echo "<div class='rating-value' style='width: " . ($averageRating * 20) . "%;'></div>";
            echo "<span class='rating-number'>" . round($averageRating, 2) . "</span>";
            echo "</div>";

            // Display the "View Review" link with the review count in brackets
            echo "<p class='more-reviews'><a href='#' class='view-all-reviews' data-product-id='" . $row['product_id'] . "'>View All Reviews (" . $reviewCount . ")</a></p>";

            echo "<div class='line-design'></div>"; // Line design

            while ($review = mysqli_fetch_assoc($reviewsResult)) {
                echo "<li class='comment'><strong>" . $review['first_name'] . ": </strong>" . $review['comment'] . "</li>";
            }

            echo "</ul>";

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

include 'liveconfig.php';
//include 'localconfig.php';



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

<script>
document.addEventListener('DOMContentLoaded', function() {
  var viewAllLinks = document.getElementsByClassName('view-all-reviews');
  Array.from(viewAllLinks).forEach(function(link) {
    link.addEventListener('click', function(event) {
      event.preventDefault();
      var productId = this.getAttribute('data-product-id');
      fetchReviews(productId);
    });
  });

  function fetchReviews(productId) {
    // Send an AJAX request to retrieve all reviews for the specified product
    // Replace 'fetch_reviews.php' with the actual endpoint URL or PHP file that handles the AJAX request
    fetch('get_reviews.php?product_id=' + productId)
      .then(function(response) {
        return response.text();
      })
      .then(function(reviewsHtml) {
        // Create a popup dialog box and display the reviews
        var dialog = document.createElement('div');
        dialog.classList.add('popup-dialog');

        var header = document.createElement('div');
        header.classList.add('popup-header');
        header.innerHTML = '<h2>View All Reviews of Product</h2>';

        var content = document.createElement('div');
        content.classList.add('popup-content');
        content.innerHTML = reviewsHtml;

        var closeButton = document.createElement('button');
        closeButton.textContent = 'Close';
        closeButton.addEventListener('click', function() {
          dialog.remove();
        });

        dialog.appendChild(header);
        dialog.appendChild(content);
        dialog.appendChild(closeButton);

        document.body.appendChild(dialog);
      })
      .catch(function(error) {
        console.log('Error retrieving reviews:', error);
      });
  }
});
</script>



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


