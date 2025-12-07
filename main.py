import cv2 as cv
import sys


def wait_then_destroy():
    if cv.waitKey(0) == ord("q"):
        sys.exit(0)
    cv.destroyAllWindows()


def process_image(image):
    # Bilateral filter
    image = cv.bilateralFilter(image, 20, 40, 40)
    cv.imshow("Bilateral filtering", image)
    wait_then_destroy()

    # Grayscale
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.imshow("Grayscale", image)
    wait_then_destroy()

    # Canny
    image = cv.Canny(image, 110, 170, L2gradient=True)
    cv.imshow("Canny", image)
    wait_then_destroy()

    # Close
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (5, 5))
    image = cv.morphologyEx(image, cv.MORPH_CLOSE, kernel, iterations=1)
    cv.imshow("Close", image)
    wait_then_destroy()

    return image


def find_contours(image):
    contours, _ = cv.findContours(
        image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return contours


def filter_by_area(contours, min_area=800):
    filtered = []
    for contour in contours:
        area = cv.contourArea(contour)
        if area > min_area:
            filtered.append(contour)
    return filtered


def main():
    IMAGE_PATH = "KPI_campus.png"
    image = cv.imread(IMAGE_PATH)
    if image is None:
        print(f"Could not read the image: {IMAGE_PATH}")
        sys.exit(1)

    # Process the image
    processed = process_image(image)

    # Find contours
    contours = find_contours(processed)
    with_contours = image.copy()
    cv.drawContours(with_contours, contours, -1, (0, 255, 0), 2)
    cv.imshow("Contours", with_contours)
    wait_then_destroy()

    # Filter contours by area
    with_filtered = image.copy()
    filtered = filter_by_area(contours)
    cv.drawContours(with_filtered, filtered, -1, (0, 255, 0), 2)
    cv.imshow("Filtered contours", with_filtered)
    print(f"Building count: {len(filtered)}")
    wait_then_destroy()


if __name__ == "__main__":
    main()
