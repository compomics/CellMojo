# -*- coding: utf-8 -*-
# generate a html report


def gen_report(exp_details):

    html_content = """
    <html lang='en' xmlns='http://www.w3.org/1999/xhtml'>
    <head>
    <meta charset='utf-8'/>
    <title>CellSET: segmentation and tracking report</title>
    <link rel='stylesheet' href='images/jquery-ui-1.10.4.custom.min.css'>
    <script src='images\jquery-1.10.2.js'></script>
    <script src='images\jquery-ui-1.10.4.custom.min.js'></script>
    <link href='http://fonts.googleapis.com/css?family=Oswald' rel='stylesheet' type='text/css'>
    <style type="text/css">
    * {
        margin: 0;
        padding: 0;
    }
    body {
        background: url(images/noise_light-grey.jpg);
        font-family: 'Helvetica Neue', arial, sans-serif;
        font-weight: 200;
    }

    h1 {
        font-family: 'Oswald', sans-serif;
        font-size: 4em;
        font-weight: 400;
        margin: 0 0 20px;
        text-align: center;
        text-shadow: 1px 1px 0 #fff, 2px 2px 0 #bbb;
    }
    hr {
        border-top: 1px solid #ccc;
        border-bottom: 1px solid #fff;
        margin: 25px 0;
        clear: both;
    }
    .centered {
        text-align: center;
    }
    .wrapper {
        width: 100%;
        padding: 30px 0;
    }
    .container {
        width: 1200px;
        margin: 0 auto;
    }
    ul.grid-nav {
        list-style: none;
        font-size: .85em;
        font-weight: 200;
        text-align: center;
    }
    ul.grid-nav li {
        display: inline-block;
    }
    ul.grid-nav li a {
        display: inline-block;
        background: #999;
        color: #fff;
        padding: 10px 20px;
        text-decoration: none;
        border-radius: 4px;
        -moz-border-radius: 4px;
        -webkit-border-radius: 4px;
    }
    ul.grid-nav li a:hover {
        background: #7b0;
    }
    ul.grid-nav li a.active {
        background: #333;
    }
    .grid-container {
        display: none;
    }
    /* ----- Image grids ----- */
    ul.rig {
        list-style: none;
        font-size: 0px;
        margin-left: -2.5%; /* should match li left margin */
    }
    ul.rig li {
        display: inline-block;
        padding: 10px;
        margin: 0 0 2.5% 2.5%;
        background: #fff;
        border: 1px solid #ddd;
        font-size: 16px;
        font-size: 1rem;
        vertical-align: top;
        box-shadow: 0 0 5px #ddd;
        box-sizing: border-box;
        -moz-box-sizing: border-box;
        -webkit-box-sizing: border-box;
    }
    ul.rig li img {
        max-width: 100%;
        height: auto;
        margin: 0 0 10px;
    }
    ul.rig li h3 {
        margin: 0 0 5px;
    }
    ul.rig li p {
        font-size: .9em;
        line-height: 1.5em;
        color: #999;
    }
    /* class for 2 columns */
    ul.rig.columns-2 li {
        width: 47.5%; /* this value + 2.5 should = 50% */
    }
    /* class for 3 columns */
    ul.rig.columns-3 li {
        width: 30.83%; /* this value + 2.5 should = 33% */
    }
    /* class for 4 columns */
    ul.rig.columns-4 li {
        width: 22.5%; /* this value + 2.5 should = 25% */
    }

    @media (max-width: 1199px) {
        .container {
            width: auto;
            padding: 0 10px;
        }
    }

    @media (max-width: 480px) {
        ul.grid-nav li {
            display: block;
            margin: 0 0 5px;
        }
        ul.grid-nav li a {
            display: block;
        }
        ul.rig {
            margin-left: 0;
        }
        ul.rig li {
            width: 100% !important; /* over-ride all li styles */
            margin: 0 0 20px;
        }
    }
    </style>
    </head>

    <body>
    <table style='margin-left:100px;'>
     <tr>
      <td width='800' height='40'>
    <div><img src='image\cropped-header_120124-011.jpg' height='37' width='151' style='margin:0px'></div>
      </td>
      <td width='800' height='40'>
    <p style= ' font-size: 1em; font-weight: normal; margin-left: 0px;'></p>
      </td>
     </tr>
     <tr>
      <td width='800' height='40'>
    <p style= ' font-size: 1em; font-weight: normal; margin-left: 0px;'>
    <p style= ' font-size: 1em; font-weight: normal; margin-left: 0px;'>
    <h3 style='margin: 10px;'> Project Title:"""+ exp_details[0] + """</h3>
    <h4 style='margin: 10px;'> Experiment Description :</b>"""+str(exp_details[1])+"""</h4>
    <h4 style='margin: 10px;'> Experiment Name :</b> """+exp_details[2]+"""</h4>
    <p style='margin: 10px;'>
     <table border=0 width=100%>

      <tr>
       <td><b>Time between images :</td> <td width='300' height='30'>"""+exp_details[3]+"""</td>
      </tr>

      <tr>
       <td><b>Smoothing technique :</td> <td width='300' height='30'>"""+exp_details[4]+""" </td>
      </tr>

      <tr>
       <td><b>Cell estimation :</td> <td width='300' height='30'>"""+exp_details[5]+""" </td>
      </tr>

      <tr>
       <td><b>Cell visibility :</td> <td width='300' height='30'>"""+exp_details[6]+""" </td>
      </tr>

      <tr>
       <td><b>Distance between cells :</td> <td width='300' height='30'> """+exp_details[7]+""" µm</td>
      </tr>

      <tr>
       <td><b>Image background color :</td> <td width='300' height='30'>"""+exp_details[8]+"""</td>
      </tr>
      <tr>
       <td><b>Minumum cell size estimate (10~500px) :</td> <td width='300' height='30'> """+exp_details[9]+""" px</td>
      </tr>

        <tr>
       <td><b>Segmentation technique :</td> <td width='300' height='30'>"""+exp_details[10]+"""</td>
      </tr>

        <tr>
       <td><b>Tracking technique :</td> <td width='300' height='30'>"""+exp_details[11]+"""</td>
      </tr>

        <tr>
       <td><b>Color:</td> <td width='300' height='30'>"""+exp_details[12]+"""</td>
      </tr>

        <tr>
       <td><b>Track no  :</td> <td width='300' height='30'>"""+exp_details[13]+"""</td>
      </tr>


      <tr>
       <td><b>Min track Duration :</td> <td width='300' height='30'> """+exp_details[14]+"""</td>
       </tr>

      <tr>
       <td><b>Culture medium :</td> <td width='300' height='30'>"""+exp_details[15]+"""</td>
      </tr>

      <tr>
        <td><b>Report generations :</td> <td width='300' height='30'> Date:"""+exp_details[16]+"""</td>
      </tr>

     </table>
    </p>
    </p>
    <p style= ' font-size: 1em; font-weight: bold; margin-left: 0px;'>Plots:Detail Reports</p>
    <p style= ' font-size: 0,8em; font-weight: normal; margin-left: 0px;'>click on a gif below to observe the correctness of the tracker</
    <img src="image\finalTrajectory.png" alt="overlayed trajectory" title="overlay trajectories" />


    <div class="wrapper">
        <div class="container">
            <h1>Detailed description of the final tracks</h1>
            <ul class="grid-nav">
                <li><a href="#" data-id="three-columns" class="active">show visualization</a></li>
                <li><a href="#" data-id="two-columns"> Detailed images</a></li>
                <!--<li><a href="#" data-id="four-columns">4 Columns</a></li>-->
            </ul>

            <hr />

            <div id="two-columns" class="grid-container" style="display:block;">
                <ul class="rig columns-2">
                    <li>
                        <img src="image\raw_image.gif" />
                        <h3>Sample of raw images</h3>
                        <p>This is a sample of raw images</p>
                    </li>
                    <li>
                        <img src="image\SegImage.gif" />
                        <h3>Sample of segmented images</h3>
                        <p>An example of applying segmentation algorithms on the raw image</p>
                    </li>
                    <li>
                        <img src="image/finalTrajectory.png" />
                        <h3>Final tracks</h3>
                        <p>The final tracks produced by the tracker</p>
                    </li>
                    <li>
                        <img src="image/animation.gif" />
                        <h3>Track movie </h3>
                        <p>Time-lapsed image sequence of the tracks</p>
                    </li>
                </ul>
            </div>
            <!--/#two-columns-->

            <div id="three-columns" class="grid-container">
                <ul class="rig columns-3">
                    <li>
                        <img src="image/raw_image.gif" />
                        <h3>Sample of raw data</h3>
                        <p>This is a sample of raw images</p>
                    </li>
                    <li>
                        <img src="image/finalTrajectory.png" />
                        <h3>Final track plot</h3>
                        <p>The final tracks produced by the tracker</p>
                    </li>
                    <li>
                        <img src="image/animation.gif" />
                        <h3>Tracking Movie</h3>
                        <p>Time-lapsed image sequence of the tracks</p>
                    </li>

                </ul>
            </div>
            <!--/#three-columns-->


            <hr />

            <p class="centered">Demo by <a href="">Sami</a></p>
        </div>
        <!--/.container-->
    </div>
    <!--/.wrapper-->

    <script src="http://ajax.googleapis.com/ajax/libs/jquery/1.10.2/jquery.min.js" type="text/javascript"></script>
    <script type="text/javascript">
    $(document).ready(function() {
        $('.grid-nav li a').on('click', function(event){
            event.preventDefault();
            $('.grid-container').fadeOut(500, function(){
                $('#' + gridID).fadeIn(500);
            });
            var gridID = $(this).attr("data-id");

            $('.grid-nav li a').removeClass("active");
            $(this).addClass("active");
        });
    });
    </script>


    </body>
    <head>

    </head>
    </html>"""
    fh = open("report3.html", "w")
    fh.write(html_content)
    fh.close()

