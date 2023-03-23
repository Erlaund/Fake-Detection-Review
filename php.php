<?php
    $message = "передай сюда свое сообщение"
    echo $message

    $array = Array("message" => $message);
    $json = json_encode($array)

    //write json to file
    if (file_put_contents("C:/xampp/htdocs/TanyaGit/SchoolCourses_back/interpretator/data.json", $json))
        echo "JSON file created successfully";
    else 
        echo "Oops! Error creating json file";

    $command = escapeshellcmd("C:/xampp/htdocs/TanyaGit/SchoolCourses_back/interpretator/python.exe 
                               C:/xampp/htdocs/TanyaGit/printSysArg.py 
                               C:/xampp/htdocs/TanyaGit/SchoolCourses_back/interpretator/data.json");
    $output = shell_exec($command);
    echo $output
?>
