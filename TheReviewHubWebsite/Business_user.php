<!DOCTYPE html>
<html lang="en">
  <head>
   
	<style>
    /* Example CSS styles for the form */

    .form-horizontal {
      max-width: 400px;
      margin: 0 auto;
      padding: 20px;
      background-color: #f2f2f2;
      border: 1px solid #ccc;
      border-radius: 5px;
    }

    .control-group {
      margin-bottom: 15px;
    }

    .control-label {
      font-weight: bold;
      display: block;
    }

    .controls {
      margin-left: 150px;
    }

    input[type="text"],
    input[type="password"] {
      width: 200px;
      padding: 5px;
      border: 1px solid #ccc;
      border-radius: 3px;
    }

    input[type="submit"] {
      padding: 10px 20px;
      background-color: #333;
      color: #fff;
      border: none;
      border-radius: 3px;
      cursor: pointer;
    }

    input[type="submit"]:hover {
      background-color: #555;
    }

    .sup {
      color: red;
    }
  </style>
  
   <style>
    /* Example CSS styles for the form */
    /* CSS styles go here */

    .sup {
      color: red;
    }
  </style>
  <script>
    function showMessage() {
      alert("Record saved successfully!");
    }
  </script>
	 
  </head>
  



<body>
<?php
include 'heade_user.php';
?>

<?php
// Establish a connection to your MySQL database

include 'liveconfig.php';
//include 'localconfig.php';



// Check the connection
if (!$conn) {
    die("Connection failed: " . mysqli_connect_error());
}

// Function to sanitize user inputs
function sanitizeInput($input)
{
    $input = trim($input);
    $input = stripslashes($input);
    $input = htmlspecialchars($input);
    return $input;
}

// Save or update record
if ($_SERVER["REQUEST_METHOD"] == "POST") {
    $business_name = sanitizeInput($_POST["inputBusinessName"]);
    $business_description = sanitizeInput($_POST["inputDescription"]);
    $city = sanitizeInput($_POST["inputcity"]);
    $contact_number = sanitizeInput($_POST["contact_number"]);
    $email = sanitizeInput($_POST["inputEmail"]);
    $website_url = sanitizeInput($_POST["website_url"]);
    $password = sanitizeInput($_POST["inputPassword"]);

    // Check if the business ID already exists
    $sql = "SELECT Business_id FROM business_registration WHERE Business_name = '$business_name'";
    $result = mysqli_query($conn, $sql);
    if (mysqli_num_rows($result) > 0) {
        // Update existing record
        $sql = "UPDATE business_registration SET Business_Description = '$business_description', City = '$city', ContactNumber = '$contact_number', Email = '$email', WebsiteURL = '$website_url', Password = '$password' WHERE Business_name = '$business_name'";
    } else {
        // Insert new record
        $sql = "INSERT INTO business_registration (Business_name, Business_Description, City, ContactNumber, Email, WebsiteURL, Password, RegistrationDate) VALUES ('$business_name', '$business_description', '$city', '$contact_number', '$email', '$website_url', '$password', NOW())";
    }

    if (mysqli_query($conn, $sql)) {
        echo "Record saved/updated successfully.";
    } else {
        echo "Error: " . $sql . "<br>" . mysqli_error($conn);
    }
}

// Close the database connection
mysqli_close($conn);
?>



<!--
Lower Header Section 
-->


	
	<form class="form-horizontal" method="POST" action="<?php echo htmlspecialchars($_SERVER["PHP_SELF"]); ?>">
    <h3>Business Registration Details</h3>

    <div class="control-group">
        <label class="control-label" for="inputBusinessName">Business name <sup class="sup">*</sup></label>
        <div class="controls">
            <input type="text" id="inputBusinessName" name="inputBusinessName" placeholder="Business name" value="<?php echo isset($business_name) ? $business_name : ''; ?>">
        </div>
    </div>

    <div class="control-group">
        <label class="control-label" for="inputDescription">Business Description <sup class="sup">*</sup></label>
        <div class="controls">
            <textarea rows="5" cols="40" id="inputDescription" name="inputDescription" placeholder="Business Description"><?php echo isset($business_description) ? $business_description : ''; ?></textarea>
        </div>
    </div>

    <div class="control-group">
        <label class="control-label" for="inputcity">City <sup class="sup">*</sup></label>
        <div class="controls">
            <input type="text" id="inputcity" name="inputcity" placeholder="City" value="<?php echo isset($city) ? $city : ''; ?>">
        </div>
    </div>

    <div class="control-group">
        <label class="control-label" for="contact_number">Contact Number <sup class="sup">*</sup></label>
        <div class="controls">
            <input type="text" id="contact_number" name="contact_number" placeholder="Contact Number" value="<?php echo isset($contact_number) ? $contact_number : ''; ?>">
        </div>
    </div>

    <div class="control-group">
        <label class="control-label" for="inputEmail">Email <sup class="sup">*</sup></label>
        <div class="controls">
            <input type="text" id="inputEmail" name="inputEmail" placeholder="Email" value="<?php echo isset($email) ? $email : ''; ?>">
        </div>
    </div>

    <div class="control-group">
        <label class="control-label" for="website_url">Website URL <sup class="sup">*</sup></label>
        <div class="controls">
            <input type="text" id="website_url" name="website_url" placeholder="Website URL" value="<?php echo isset($website_url) ? $website_url : ''; ?>">
        </div>
    </div>

    <div class="control-group">
        <label class="control-label" for="inputPassword">Password <sup class="sup">*</sup></label>
        <div class="controls">
            <input type="password" id="inputPassword" name="inputPassword" placeholder="Password">
        </div>
    </div>

    <div class="control-group">
        <div class="controls">
            <input type="submit" name="submitAccount" value="Submit" class="exclusive shopBtn">
        </div>
    </div>
</form>

</div>





</div>
</div>
<!-- 
Clients 
-->


<!--
Footer
-->

  </body>
</html>
<script>
// JavaScript function to validate the date format
function validateDateFormat(input) {
  // Regular expression to match the "dd/mm/yyyy" format
  var regex = /^\d{2}\/\d{2}\/\d{4}$/;
  
  if (!regex.test(input.value)) {
    alert("Invalid date format. Please enter the date in dd/mm/yyyy format.");
    input.value = ""; // Clear the input field
    input.focus();    // Set focus back to the input field
  }
}

// Event listener to validate the date format when the input loses focus
var dateInput = document.getElementById("dateInput");
dateInput.addEventListener("blur", function() {
  validateDateFormat(this);
});
</script>