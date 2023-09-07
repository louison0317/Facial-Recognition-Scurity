# Facial-Recognition-Scurity
This is a comprehensive and practical facial recognition access control system, include visitor identification, stranger alert and real-time notification.


<!-- ABOUT THE PROJECT -->
## About The Project

* Visitor Identification: The system is capable of identifying visitors and distinguishing whether they are teachers, students, parents, or strangers.

* Notification Feature: For different identified individuals, the system will send corresponding notifications. For example:

> When teachers enter or leave the campus, the system will automatically record the entry and exit times and notify relevant personnel.

> When students enter or leave the campus, the system will similarly record the times and notify relevant personnel.

> When parents pick up their children, the system will notify the respective teachers.

* Stranger Alert: If the system identifies a stranger, it will display a warning on the screen and send an alert to all teachers.

> Incident of Student Being Taken Away: If a stranger takes a student away, the system will save a photo of the stranger and notify the student's parents.

* Technological Applications:

1. `Facial Feature Extraction`: Utilizing dlib to extract facial features, which is an advanced facial recognition library.
2. `Image Processing`: Using OpenCV for image processing, which is a powerful open-source computer vision library.
3. `Website Construction`: Building a membership website with Flask, a clean and efficient Python web framework.
4. `Data Storage`: Using SQLite for data storage, a lightweight relational database management system.
5. `Real-Time Notification`: Integrating with Linebot, a popular communication software in Taiwan, allows for real-time notifications.

* Data Management: Actions such as querying, adding, and deleting records can be performed on the Linebot, enhancing the system's controllability and convenience.


<!-- GETTING STARTED -->
## Getting Started

### Installation

_Below is an example of how you can instruct your audience on installing and setting up your app._

1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo
   ```sh
   git clone https://github.com/your_username_/Project-Name.git
   ```
3. Install NPM packages
   ```sh
   npm install
   ```
4. Enter your API in `config.js`
   ```js
   const API_KEY = 'ENTER YOUR API';
   ```
