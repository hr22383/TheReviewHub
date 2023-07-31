<?php
include 'liveconfig.php';
//include 'localconfig.php';


session_start();

// Access the Business ID stored in the session
$businessId = $_SESSION['Business_id'];

try {
    // Query to fetch the business owner's data
    $query = "SELECT * FROM business_registration WHERE Business_id = $businessId";
    $result = mysqli_query($conn, $query);

    if ($result) {
        $businessOwner = mysqli_fetch_assoc($result);
        mysqli_free_result($result); // Free the result set
    } else {
        die("Error executing the query: " . mysqli_error($conn));
    }

} catch (mysqli_sql_exception $e) {
    // Handle any database errors here
    die("Database Error: " . $e->getMessage());
}
?>

<!DOCTYPE html>
<html>
<head>
    <title>Business Owner Profile</title>
    <style>
        /* CSS styling for the profile layout */
        .profile-container {
            max-width: 600px;
            margin: 0 auto;
            padding: 20px;
            border: 1px solid #ccc;
            border-radius: 5px;
            background-color: #f9f9f9;
        }

        .profile-heading {
            text-align: center;
            margin-bottom: 20px;
        }

        table {
            width: 100%;
        }

        th, td {
            padding: 10px;
            text-align: left;
        }

        th {
            background-color: #f2f2f2;
        }
    </style>
</head>
<body>

<?php
include 'header.php';
?>
    <div class="profile-container">
        <h1 class="profile-heading">Business Owner Profile</h1>

        <table>
            <?php foreach ($businessOwner as $field => $value) { ?>
                <tr>
                    <td><?php echo ucfirst(str_replace('_', ' ', $field)); ?>:</td>
                    <td><?php echo $value; ?></td>
                </tr>
            <?php } ?>
        </table>
    </div>
</body>
</html>
