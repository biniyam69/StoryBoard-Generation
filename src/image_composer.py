from typing import List, Literal, Tuple
import itertools
import random
from collections import defaultdict

from PIL import Image
from pprint import pprint


VERTICAL_POSITIONING = {'Logo': [1], 'CTA Button': [1, 2, 3], 'Icon': [1, 2, 3], 'Product Image': [2],
               'Text Elements': [1,3], 'Infographic': [2], 'Banner': [1], 'Illustration': [2], 'Photograph': [2],
               'Mascot': [2], 'Testimonial Quotes': [2], 'Social Proof': [2, 1, 3], 'Seal or Badge': [3, 1, 2],
               'Graphs and Charts': [2], 'Decorative Elements': [3], 'Interactive Elements': [2],
               'Animation': [2], 'Coupon or Offer Code': [3], 'Legal Disclaimers or Terms': [3],
               'Contact Information': [3, 1, 2], 'Map or Location Image': [3], 'QR Code': [3, 1, 2]}

HORIZONTAL_POSITIONING = {'Logo': [1], 'CTA Button': [2, 1, 3], 'Icon': [1], 'Product Image': [1],
                          'Text Elements': [1], 'Infographic': [1], 'Banner': [2], 'Illustration': [2],
                          'Photograph': [2], 'Mascot': [1], 'Testimonial Quotes': [2], 'Social Proof': [3, 1, 2],
                          'Seal or Badge': [3, 1, 2], 'Graphs and Charts': [1], 'Decorative Elements': [3],
                          'Interactive Elements': [2], 'Animation': [2], 'Coupon or Offer Code': [3],
                          'Legal Disclaimers or Terms': [3], 'Contact Information': [3, 1, 2],
                          'Map or Location Image': [3], 'QR Code': [3, 1, 2]}


class ImageComposer:
    categories = Literal["Background", "Logo", "CTA Button", "Icon", "Product Image", "Text Elements", "Infographic", "Banner", "Illustration", "Photograph", "Mascot", "Testimonial Quotes", "Social Proof", "Seal or Badge", "Graphs and Charts", "Decorative Elements", "Interactive Elements", "Animation", "Coupon or Offer Code", "Legal Disclaimers or Terms", "Contact Information", "Map or Location Image", "QR Code"]
    PositionSegment = Tuple[float, float]
    AlignmentPosition = Tuple[int, int]
    AlignmentPositions = List[AlignmentPosition]
    frame_images = List[Tuple[categories, str, str]]

    def __init__(self, width:int, height: int, frames: List[frame_images]) -> None:
        self.width = width
        self.height = height
        self.frames = frames
        self.segments = ImageComposer.get_image_position_segments(width, height)
        self.generated_frames = []

    def generate_frames(self):
        self.compose_frames()
        return self.generated_frames

    def compose_frames(self) -> None:
        self.generated_frames = []

        for frame in self.frames:
            # Separate Background
            placement_items = []
            for index, item in enumerate(frame):
                if item[0] == "Background":
                    background_index = index
                    continue
                placement_items.append(item)
            
            background = frame[background_index]

            possibilties = ImageComposer.compute_positions([item[0] for item in placement_items])
            identified_locations = ImageComposer.select_diverse_positions(possibilties)
            adjusted_positions = self.calculate_adjusted_element_positions(identified_locations)
            placement_values = [(x[2], *list(y.values())) for x, y in zip(placement_items, adjusted_positions)]
            # Construct Frame
            self.generated_frames.append(self.create_combined_image(background[2], placement_values))

    @staticmethod
    def compute_positions(elements: List[categories]) -> List[AlignmentPositions]:
        possible_positions = []

        # Iterate through each element to calculate its position combinations
        for element in elements:
            vertical_options = VERTICAL_POSITIONING[element]
            horizontal_options = HORIZONTAL_POSITIONING[element]
            combinations = list(itertools.product(vertical_options, horizontal_options))
            possible_positions.append(combinations)

        return possible_positions
    
    @staticmethod
    def select_diverse_positions(possible_positions: List[AlignmentPositions]) -> AlignmentPositions:
        position_frequency = defaultdict(int)

        def update_position_frequency(selected_position):
            position_frequency[selected_position] += 1

        selected_positions = []
        
        for positions in possible_positions:
            sorted_combinations = sorted(positions, key=lambda x: position_frequency[x])
            
            lowest_frequency = position_frequency[sorted_combinations[0]]
            lowest_freq_combinations = [pos for pos in sorted_combinations if position_frequency[pos] == lowest_frequency]
            
            selected_position = random.choice(lowest_freq_combinations)
            selected_positions.append(selected_position)
            
            update_position_frequency(selected_position)
        
        return selected_positions


    @staticmethod
    def get_image_position_segments(width: float, height: float, vm: float = 0.6, vo: float = 0.2, hm: float = 0.6, ho: float = 0.2) -> Tuple[List[PositionSegment], List[PositionSegment]]:
        """Divide Image based on percentage for vertical and horizontal segments."""
        
        if vm + vo * 2 > 1 or hm + ho * 2 > 1:
            raise ValueError("Sum of percentages exceeds 100% for either vertical or horizontal segments.")
        
        vertical_mid = height * vm
        vertical_outer = height * vo
        horizontal_mid = width * hm
        horizontal_outer = width * ho

        vertical_segments = [
            (0, vertical_outer),
            (vertical_outer, vertical_outer + vertical_mid),
            (vertical_outer + vertical_mid, height)
        ]
        
        horizontal_segments = [
            (0, horizontal_outer),
            (horizontal_outer, horizontal_outer + horizontal_mid),
            (horizontal_outer + horizontal_mid, width)
        ]

        segements = []
        for vs in vertical_segments:
            vs_items = []
            for hs in horizontal_segments:
                vs_items.append((vs, hs))
            segements.append(vs_items)


        return segements
    
    def calculate_adjusted_element_positions(self, elements_positions, padding=10):
        element_details = []
        segment_elements = {}

        # Organize elements by their segments
        for i, (v_pos, h_pos) in enumerate(elements_positions):
            segment_key = (v_pos, h_pos)
            if segment_key not in segment_elements:
                segment_elements[segment_key] = []
            segment_elements[segment_key].append(i)
        
        for segment_key, elements in segment_elements.items():
            v_pos, h_pos = segment_key
            segment = self.segments[v_pos-1][h_pos-1]
            vertical_segment, horizontal_segment = segment
            num_elements = len(elements)
            
            x_start, x_end = horizontal_segment
            y_start, y_end = vertical_segment
            segment_width = (x_end - x_start) - 2 * padding
            segment_height = (y_end - y_start) - 2 * padding
            
            # Determine alignment and divide space
            is_vertical = segment_height > segment_width
            if is_vertical:
                space_per_element = segment_height / num_elements
            else:
                space_per_element = segment_width / num_elements
            
            for index, _ in enumerate(elements):
                if is_vertical:
                    element_x_start = x_start + padding
                    element_y_start = y_start + padding + index * space_per_element
                    element_width = segment_width
                    element_height = space_per_element
                else:
                    element_x_start = x_start + padding + index * space_per_element
                    element_y_start = y_start + padding
                    element_width = space_per_element
                    element_height = segment_height
                
                element_details.append({
                    "start_point": (element_x_start, element_y_start),
                    "dimensions": (element_width, element_height)
                })

        return element_details
    
    @staticmethod
    def resize_image(image, target_width, target_height):
        """
        Resize an image to fit within target dimensions while maintaining aspect ratio.
        """
        original_width, original_height = image.size
        ratio = min(target_width / original_width, target_height / original_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)
        resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)
        return resized_image

    def create_combined_image(self, background_path: str, elements: List[Tuple[str, int|float, int|float]]) -> Image.Image:
        """
        Create a combined image based on background and elements' positioning and sizing.
        
        :param background_path: Path to the background image.
        :param elements: A list of dictionaries, each containing 'image_path', 'start_point', and 'dimensions'.
        """
        # Load the background image
        background = Image.open(background_path).convert("RGBA")
        
        for element in elements:
            # Load element image
            image_path = element[0]
            image = Image.open(image_path).convert("RGBA")
            
            # Resize image according to dimensions without losing aspect ratio
            target_width, target_height = element[2]
            resized_image = ImageComposer.resize_image(image, target_width, target_height)
            
            # Calculate position to center the image within its segment
            start_x, start_y = element[1]
            offset_x = start_x + (target_width - resized_image.size[0]) / 2
            offset_y = start_y + (target_height - resized_image.size[1]) / 2
            
            # Place the resized image on the background
            background.paste(resized_image, (int(offset_x), int(offset_y)), resized_image)
        
        return background
    

if __name__ == "__main__":
    ic = ImageComposer(320, 500, [[('Logo', 'url_path', 'local_path'), 
                                   ('Call-To-Action (CTA) Button', 'url_path', 'local_path'),
                                   ('Icon', 'url_path', 'local_path'),
                                   ('Product Image', 'url_path', 'local_path'),
                                   ('Text Elements', 'url_path', 'local_path')]])
    possibilties = ImageComposer.compute_positions(["Logo", "Call-To-Action (CTA) Button", "Icon", "Product Image", "Text Elements"])
    pprint(possibilties)
    print("======================================================")
    diverse = ImageComposer.select_diverse_positions(possibilties)
    pprint(diverse)

    print(ic.calculate_adjusted_element_positions(diverse))
    