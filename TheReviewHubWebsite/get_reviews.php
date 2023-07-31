<!DOCTYPE html>
<html>
<head>
  <style>
    .highlight-dash {
      color: red; /* Set the color to red */
      font-weight: bold; /* Make it bold */
      font-size: 16px; /* Increase the font size */
      /* Add more styles as per your preference */
    }
  </style>
</head>
<body>
  <?php
  //include 'liveconfig.php';
   include 'localconfig.php';

  // Retrieve the product ID from the AJAX request
  $productId = $_GET['product_id'];

  // Retrieve all reviews for the specified product
  $reviewsSql = "SELECT * FROM reviews WHERE product_id = $productId";
  $reviewsResult = mysqli_query($conn, $reviewsSql);

  if (mysqli_num_rows($reviewsResult) > 0) {
    echo "<ul class='comment-list'>";
    while ($review = mysqli_fetch_assoc($reviewsResult)) {
      echo "<li class='comment'>";
      echo "<span class='reviewer-name'>";
      
      // Highlight the reviewer's name with dashes
      $reviewerName = $review['user_name'];
      $highlightedName = "";
      for ($i = 0; $i < strlen($reviewerName); $i++) {
        $highlightedName .= "<span class='highlight-dash'>" . $reviewerName[$i] . "</span>";
      }
      echo "<span style='font-weight: bold; font-size: 20px;'>{$highlightedName}</span>";
      
      echo ":</span> ";
      echo $review['comment'];
      echo "</li>";
    }
    echo "</ul>";
  } else {
    echo "<p class='no-reviews'>No reviews found for this product.</p>";
  }

  mysqli_close($conn);
  ?>
</body>
</html>
