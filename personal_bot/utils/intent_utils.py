"""
This module defines the core intent classes used for handling different types of user requests in the chat system.
Each intent class represents a specific type of user request (dining, travel, cab booking, gifting) and manages
the relevant information associated with that intent.
"""

from typing import Optional, List

class Intent:
    """
    Base class for all intent types. Provides common functionality for managing intent information.
    
    This class implements the basic operations that all intent types should support:
    - Getting all keys (attributes) of the intent
    - Identifying missing information
    - Updating intent information
    - Retrieving current intent information
    """
    
    def get_keys(self):
        """Returns a list of all attribute names (keys) defined in the intent class."""
        return [key for key in self.__dict__]
    
    def get_missing_info(self):
        """Returns a list of attribute names that have not been set (are None)."""
        return [key for key in self.__dict__ if getattr(self, key) is None]
    
    def update_info(self, info: dict):
        """
        Updates the intent's attributes with new information.
        
        Args:
            info (dict): Dictionary containing key-value pairs to update the intent's attributes
        """
        for key, value in info.items():
            setattr(self, key, value)

    def get_updated_info(self):
        """Returns a dictionary of all non-None attributes and their values."""
        return {key: getattr(self, key) for key in self.__dict__ if getattr(self, key) is not None}
    
    def get_info(self):
        """Returns a dictionary of all attributes and their values, including None values."""
        return {key: getattr(self, key) for key in self.__dict__}


class DiningIntent(Intent):
    """
    Represents the dining intent, handling restaurant reservations and dining queries.
    
    Attributes:
        date (Optional[str]): Date for the dining reservation
        time (Optional[str]): Time for the dining reservation
        location (Optional[str]): Location/area for dining
        budget (Optional[str]): Budget range for dining
        cuisine (Optional[str]): Preferred cuisine type
        party_size (Optional[str]): Number of people dining
        special_requests (Optional[List[str]]): Any special requirements or preferences
    """
    def __init__(self):
        self.date: Optional[str] = None
        self.time: Optional[str] = None
        self.location: Optional[str] = None
        self.budget: Optional[str] = None
        self.cuisine: Optional[str] = None
        self.party_size: Optional[str] = None
        self.special_requests: Optional[List[str]] = None
    
class TravelIntent(Intent):
    """
    Represents the travel intent, handling trip planning and travel arrangements.
    
    Attributes:
        location_from (Optional[str]): Starting location of the journey
        location_to (Optional[str]): Destination of the journey
        start_date (Optional[str]): Journey start date
        end_date (Optional[str]): Journey end date
        mode (Optional[str]): Mode of travel (flight, train, etc.)
        members (Optional[str]): Number of travelers
        budget (Optional[str]): Travel budget
        special_requests (Optional[List[str]]): Any special travel requirements
    """
    def __init__(self):
        self.location_from: Optional[str] = None
        self.location_to: Optional[str] = None
        self.start_date: Optional[str] = None
        self.end_date: Optional[str] = None
        self.mode: Optional[str] = None
        self.members: Optional[str] = None
        self.budget: Optional[str] = None
        self.special_requests: Optional[List[str]] = None

class CabIntent(Intent):
    """
    Represents the cab booking intent, handling ride requests and transportation.
    
    Attributes:
        pickup_location (Optional[str]): Starting point for the ride
        drop_off_location (Optional[str]): Destination for the ride
        members (Optional[str]): Number of passengers
        budget (Optional[str]): Budget for the ride
        special_requests (Optional[List[str]]): Any special requirements for the ride
    """
    def __init__(self):
        self.pickup_location: Optional[str] = None
        self.drop_off_location: Optional[str] = None
        self.members: Optional[str] = None
        self.budget: Optional[str] = None
        self.special_requests: Optional[List[str]] = None

class GiftingIntent(Intent):
    """
    Represents the gifting intent, handling gift-related queries and requests.
    
    Attributes:
        recipient (Optional[str]): Person receiving the gift
        occasion (Optional[str]): Occasion for the gift
        budget (Optional[str]): Budget for the gift
        special_requests (Optional[List[str]]): Any special requirements for the gift
    """
    def __init__(self):
        self.recipient: Optional[str] = None
        self.occasion: Optional[str] = None
        self.budget: Optional[str] = None
        self.special_requests: Optional[List[str]] = None

if __name__ == "__main__":
    # Example usage and testing of intent classes
    dining_intent = DiningIntent()
    print(dining_intent.get_keys())
    # print(dining_intent.get_missing_info())
    # dining_intent.update_info({"date": "2024-01-01", "time": "12:00 pm"})
    # print(dining_intent.get_missing_info())
    # print(dining_intent.get_info())

    travel_intent = TravelIntent()
    print(travel_intent.get_keys())
    # print(travel_intent.get_missing_info())
    # travel_intent.update_info({"date": "2024-01-01", "location_from": "mumbai", "location_to": "delhi", "mode": "flight", "budget": 10000})
    # print(travel_intent.get_missing_info())
    # print(travel_intent.get_info())

    cab_intent = CabIntent()
    print(cab_intent.get_keys())
    # print(cab_intent.get_missing_info())
    # cab_intent.update_info({"pickup_location": "mumbai", "drop_off_location": "delhi", "members": 2, "budget": 10000})
    # print(cab_intent.get_missing_info())
    # print(cab_intent.get_info())
    
    gifting_intent = GiftingIntent()
    print(gifting_intent.get_keys())
    # print(gifting_intent.get_missing_info())
    # gifting_intent.update_info({"recipient": "mumbai", "occasion": "delhi", "budget": 10000})
    # print(gifting_intent.get_missing_info())
    # print(gifting_intent.get_info())