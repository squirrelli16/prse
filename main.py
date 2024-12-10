from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from underthesea import word_tokenize
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from waitress import serve
import logging
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import mysql.connector
import numpy as np
import random
app = Flask(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        # Log to console
        logging.StreamHandler(sys.stdout),
        # Log to file
        logging.FileHandler('chatbot.log')
    ]
)
logger = logging.getLogger(__name__)
CORS(app)


def preprocess_vietnamese(text):
    tokens = word_tokenize(text)
    return ' '.join(tokens)


def format_price(price):
    return "{:,.0f}".format(price).replace(',', '.')


def get_effective_price(row, courses_discount):
    """
    Get the effective price considering discounts
    """
    course_id = row['id']
    original_price = row['original_price']

    # Check if course has an active discount
    discount_info = courses_discount[
        (courses_discount['course_id'] == course_id) &
        (courses_discount['is_active'] == True)
        ]

    if not discount_info.empty:
        return discount_info.iloc[0]['discount_price']
    return original_price


def calculate_combined_similarity(courses, courses_discount, text_weight=0.6, rating_weight=0.2, price_weight=0.2,
                                  category_weight=0.9):
    """
    Calculate similarity between courses based on description, rating, and effective price

    Parameters:
    - courses: DataFrame containing 'id', 'description', 'average_rating', and 'original_price'
    - courses_discount: DataFrame containing discount information
    - text_weight: weight for description similarity (default: 0.6)
    - rating_weight: weight for rating similarity (default: 0.2)
    - price_weight: weight for price similarity (default: 0.2)

    Returns:
    - combined similarity matrix
    """
    # Preprocess description
    courses['description'] = courses['description'].apply(preprocess_vietnamese)

    # Calculate text similarity
    tfidf = TfidfVectorizer(
        max_features=3522,
        ngram_range=(1, 2)
    )
    text_vectors = tfidf.fit_transform(courses['description'])
    text_similarity = cosine_similarity(text_vectors)

    # Calculate effective prices considering discounts
    courses['effective_price'] = courses.apply(
        lambda row: get_effective_price(row, courses_discount),
        axis=1
    )
    # Normalize numerical features
    scaler = MinMaxScaler()

    # Handle rating similarity
    ratings = courses[['average_rating']].values
    normalized_ratings = scaler.fit_transform(ratings)
    rating_similarity = 1 - np.abs(normalized_ratings - normalized_ratings.T)

    # Handle price similarity using effective prices
    prices = courses[['effective_price']].values
    normalized_prices = scaler.fit_transform(prices)
    price_similarity = 1 - np.abs(normalized_prices - normalized_prices.T)

    # Handle category similarity: 1 if categories match, 0 if they don't
    categories = courses[['sub_category_id']].values
    category_similarity = np.array([[1 if cat1 == cat2 else 0 for cat2 in categories] for cat1 in categories])

    # Combine all similarities with their respective weights
    combined_similarity = (
            text_weight * text_similarity +
            rating_weight * rating_similarity +
            price_weight * price_similarity +
            category_weight * category_similarity
    )

    return combined_similarity


# Load data

# courses_discount = pd.read_csv('drive/MyDrive/Data_PRSE/course_discount.csv')
# courses_category = pd.read_csv('drive/MyDrive/Data_PRSE/course_category.csv')
# # Select required columns
# courses = courses[['id', 'average_rating', 'description', 'original_price']]
# courses = courses.merge(courses_category[['course_id', 'sub_category_id']], left_on='id', right_on='course_id',
#                         how='left')

# Load PhoBERT model and tokenizer
model_name = 'vinai/phobert-base'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


# Sử dụng SQLAlchemy để dễ dàng chuyển đổi sang DataFrame
def get_course():
    connection = mysql.connector.connect(
        host='14.225.253.200',  # IP address of your DB instance
        user='root',
        password='Hoanganh123!@#',
        database='prse'
    )
    cursor = connection.cursor()

    # Thay đổi câu query theo cấu trúc database của bạn
    query = """
      SELECT id, title, short_description, original_price
      FROM course
      WHERE is_publish = 1
  """

    cursor.execute(query)

    # Fetch all results
    data = cursor.fetchall()

    # Get column names
    columns = [desc[0] for desc in cursor.description]

    # Convert to DataFrame
    df = pd.DataFrame(data, columns=columns)

    # Print the DataFrame
    df['name'] = df['title']
    df['title'] = df['title'].str.lower() + ". " + df['short_description'].str.lower()
    return df

courses = get_course()
# Function to generate embeddings from PhoBERT
def generate_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
    return embeddings.squeeze().numpy()


# Function to find courses based on user input
def find_course(user_input, max_results=3):
    df = get_course()
    user_input = user_input.lower()
    user_input_embedding = generate_embedding(user_input)

    course_titles = df['title'].tolist()
    course_embeddings = np.array([generate_embedding(title) for title in course_titles])

    cosine_scores = np.dot(course_embeddings, user_input_embedding) / (
                np.linalg.norm(course_embeddings, axis=1) * np.linalg.norm(user_input_embedding))

    top_results = np.argsort(cosine_scores)[-max_results:][::-1]

    valid_results = [i for i in top_results if cosine_scores[i] >= 0.55]

    if not valid_results:
        return "Không tìm thấy khóa học phù hợp."
    else:
        # Retrieve courses that pass the cosine score threshold
        top_courses = df.iloc[valid_results]
        return top_courses


# Select required columns


# Calculate similarity matrix with effective prices
# similarity_matrix = calculate_combined_similarity(
#     courses,
#     courses_discount,
#     text_weight=0.75,
#     rating_weight=0.5,
#     price_weight=0.4
# )


# Assuming you have these loaded from somewhere
# You might want to load these when the application starts
def load_data():
    """
    Load the necessary data for recommendations
    You'll need to implement this based on how you store your data
    """
    global courses, similarity_matrix
    # Load your courses DataFrame
    # Load your similarity matrix
    pass


def get_top_rated_courses(n=10):
    """
    Get top N courses with highest ratings
    """
    try:
        # Sort by rating and get top N courses
        top_courses = courses.nlargest(n, 'average_rating')[['id', 'average_rating']]
        top_courses['based_on_courses'] = 'top_rated'  # Indicate these are top rated courses

        return top_courses
    except Exception as e:
        print(f"Error getting top rated courses: {e}")
        return None


@app.route('/api/recommend', methods=['POST'])
def get_recommendations():
    try:
        data = request.get_json()
        course_ids = data.get('course_ids', [])

        if not course_ids:
            top_rated = get_top_rated_courses(10)
            if top_rated is None:
                return jsonify({
                    'error': 'Could not fetch recommendations or top rated courses'
                }), 500

            result = top_rated.to_dict(orient='records')
            return jsonify({
                'status': 'success',
                'recommendations': result,
                'recommendation_type': 'top_rated'
            })

        # Get recommendations
        recommendations = get_similar_courses_combined(course_ids)

        if recommendations is None:
            # Get top rated courses instead
            top_rated = get_top_rated_courses(10)
            if top_rated is None:
                return jsonify({
                    'error': 'Could not fetch recommendations or top rated courses'
                }), 500

            result = top_rated.to_dict(orient='records')
            return jsonify({
                'status': 'success',
                'recommendations': result,
                'recommendation_type': 'top_rated'
            })

        # Convert DataFrame to dictionary for JSON response
        result = recommendations.to_dict(orient='records')

        return jsonify({
            'status': 'success',
            'recommendations': result,
            'recommendation_type': 'similar'
        })

    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500


def get_similar_courses_combined(course_ids, n=10):
    """
    Your existing recommendation function
    """
    try:
        # Get indices for all input course IDs
        indices = [courses.index[courses['id'] == cid][0] for cid in course_ids]

        # Extract vectors for each course
        course_vectors = similarity_matrix[indices]

        # Combine vectors by taking the mean
        combined_vector = np.mean(course_vectors, axis=0)

        # Calculate similarity between combined vector and all courses
        similarities = cosine_similarity([combined_vector], similarity_matrix)[0]

        # Get indices of top N similar courses
        # Exclude the input courses themselves
        mask = np.ones(len(similarities), dtype=bool)
        mask[indices] = False
        filtered_similarities = similarities * mask

        similar_indices = filtered_similarities.argsort()[::-1][:n]

        # Create DataFrame with similar courses
        similar_courses = courses.iloc[similar_indices][['id']]

        # Add original course IDs for reference

        return similar_courses

    except IndexError as e:
        print(f"Error: One or more course IDs not found in dataset")
        return None


# Add some error handlers
@app.errorhandler(404)
def not_found_error(error):
    return jsonify({
        'error': 'Resource not found'
    }), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error'
    }), 500


# Add a health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'message': 'Service is running'
    })


@app.route('/rec_chatbot', methods=['POST'])
def find_courses():
    data = request.get_json()
    user_input = data.get('message', '')
    logger.info(f"=== New Request ===")
    logger.info(f"User input: {user_input}")
    # Validate input
    if not user_input:
        return jsonify({'error': 'No user input provided'}), 400

    matching_courses = find_course(user_input)

    if isinstance(matching_courses, pd.DataFrame) and not matching_courses.empty:
        
        footer_responses = [
            "🌟 Hãy cùng khám phá và bắt đầu hành trình học tập thú vị với EasyEdu nhé!",
            "📚 Chúng ta hãy bước vào thế giới kiến thức đầy màu sắc cùng EasyEdu nào!",
            "🚀 Đừng ngần ngại, hãy cùng EasyEdu chinh phục những điều mới mẻ nhé!",
            "🤝 Hãy để EasyEdu đồng hành cùng bạn trong cuộc hành trình học tập tuyệt vời này!",
            "🌈 Khám phá những điều tuyệt vời và bắt đầu học cùng EasyEdu nào!",
            "✨ Chúng mình hãy cùng nhau khám phá hành trình học tập thú vị với EasyEdu nhé!",
            "📖 Hãy tham gia cùng EasyEdu để trải nghiệm những bài học bổ ích nào!",
            "🌟 Khởi đầu hành trình học tập của bạn với EasyEdu ngay hôm nay!",
            "💡 Hãy để EasyEdu giúp bạn mở rộng kiến thức và kỹ năng nhé!",
            "🎉 Chúng ta hãy cùng nhau học hỏi và phát triển với EasyEdu nào!",
            "🌍 Hãy khám phá thế giới học tập rộng lớn cùng EasyEdu nhé!",
            "🌞 Chúng ta hãy cùng EasyEdu xây dựng tương lai tươi sáng hơn!",
            "📅 Hãy bắt đầu hành trình học tập của bạn với những khóa học thú vị từ EasyEdu!",
            "🎈 Khám phá và trải nghiệm những điều mới mẻ cùng EasyEdu nào!",
            "🗺️ Hãy để EasyEdu dẫn dắt bạn trên con đường tri thức!",
            "🎓 Cùng EasyEdu tạo nên những kỷ niệm học tập đáng nhớ nhé!",
            "📝 Khám phá những khóa học hấp dẫn và thú vị với EasyEdu!",
            "🔍 Hãy tham gia cùng EasyEdu để khám phá những kiến thức mới mẻ!",
            "🏆 Chúng ta hãy cùng nhau vươn tới những đỉnh cao tri thức cùng EasyEdu!",
            "🌼 Hãy bắt tay vào hành trình học tập đầy thú vị với EasyEdu nào!"
        ]
        header_responses = [
            "🌟 Chúng tôi đã tìm thấy những khóa học thú vị đang chờ bạn! Hãy xem ngay nhé:",
            "🎉 Trợ lý EasyEdu đã phát hiện ra các khóa học độc đáo dành cho bạn. Hãy cùng khám phá nào:",
            "📚 Trợ lý EasyEdu rất vui khi tìm thấy các khóa học phù hợp với bạn! Hãy thử ngay nhé:",
            "✨ Wow! Những khóa học tuyệt vời đã sẵn sàng cho bạn. Hãy xem ngay nào:",
            "🚀 Trợ lý EasyEdu đã tìm ra các khóa học phù hợp với sở thích của bạn. Hãy khám phá nhé:",
            "🌼 Trợ lý EasyEdu rất vui thông báo rằng đã có những khóa học tuyệt vời cho bạn. Hãy xem ngay:",
            "🎈 Hooray! Các khóa học lý tưởng đang chờ đón bạn. Hãy tìm hiểu ngay nhé:",
            "😍 Trợ lý EasyEdu đã phát hiện ra những khóa học đặc biệt dành cho bạn. Hãy cùng xem nào:",
            "💖 Những khóa học phù hợp đã được tìm thấy! Hãy khám phá cùng trợ lý EasyEdu nhé:",
            "🌈 Trợ lý EasyEdu rất vui khi thông báo rằng đã có các khóa học phù hợp với bạn. Hãy xem ngay:",
            "🥳 Trợ lý EasyEdu đã tìm thấy những khóa học thú vị dành cho bạn. Hãy cùng khám phá nhé:",
            "🌟 Tốt quá! Trợ lý EasyEdu đã phát hiện ra các khóa học tuyệt vời cho bạn. Hãy xem ngay:",
            "🎉 Những khóa học phù hợp với bạn đã được tìm thấy! Hãy khám phá ngay nhé:",
            "📚 Trợ lý EasyEdu rất vui khi thông báo rằng đã có các khóa học phù hợp với bạn. Hãy xem ngay:",
            "🚀 Các khóa học lý tưởng đang chờ bạn! Hãy cùng khám phá nhé:",
            "✨ Trợ lý EasyEdu đã tìm thấy những khóa học tuyệt vời cho bạn. Hãy xem ngay nào:",
            "🌼 Hooray! Những khóa học phù hợp đã sẵn sàng cho bạn. Hãy khám phá ngay nhé:",
            "🎈 Trợ lý EasyEdu rất vui khi tìm thấy các khóa học thú vị cho bạn. Hãy cùng xem nào:",
            "😍 Trợ lý EasyEdu đã phát hiện ra những khóa học tuyệt vời dành cho bạn. Hãy khám phá ngay nhé:",
            "💖 Những khóa học phù hợp với bạn đã sẵn sàng! Hãy cùng khám phá với trợ lý EasyEdu nhé:"
        ]
        random_header = random.choice(header_responses)
        random_footer = random.choice(footer_responses)
        course_list = []
        count = 0
        for index, course_info in matching_courses.iterrows():
            course_name = course_info['name']
            course_link = 'https://prse-fe.vercel.app/course-detail/' + str(course_info['id'])
            course_price = course_info['original_price']
            course_price = format_price(course_price)

            # · Giá :
            # · Link :

            course_list.append(
                f"**{count + 1}. {course_name}**:" + "\n " + f"· **Giá :** {course_price} VND" + "\n" + f"· **Link :** {course_link}")
            count += 1
        all_courses = f"{random_header}\n" + "\n".join(course_list) + f"\n{random_footer}"
        return jsonify({'error_message': {},
                        'code': 1,
                        'data': {
                            'message': all_courses
                        }}), 200

    else:
        no_responses = [
            "🌼 Ôi không! Trợ lý EasyEdu không tìm thấy khóa học nào phù hợp với bạn. Hãy thử lại lần nữa nhé, có thể điều gì đó tuyệt vời sẽ xuất hiện!",
            "🤗 Trợ lý EasyEdu rất tiếc, nhưng hiện tại chưa tìm thấy khóa học nào hợp với bạn. Đừng ngại thử lại sau một chút nhé!",
            "💔 Rất tiếc, nhưng có vẻ như trợ lý EasyEdu chưa tìm thấy khóa học nào phù hợp. Hãy thử lại lần nữa, có thể bạn sẽ tìm thấy điều mình thích!",
            "🌟 Dù trợ lý EasyEdu chưa tìm thấy khóa học nào phù hợp, hãy thử lại nhé! Có thể điều bất ngờ đang chờ bạn ở lần sau!",
            "✨ Trợ lý EasyEdu không tìm thấy khóa học nào phù hợp với bạn. Nhưng đừng nản lòng, hãy thử lại lần nữa nhé!",
            "🌈 Trợ lý EasyEdu muốn giúp bạn, nhưng hiện tại chưa có khóa học nào phù hợp. Hãy quay lại và thử lại một lần nữa nhé!",
            "🤔 Có vẻ như trợ lý EasyEdu chưa tìm thấy khóa học nào hợp với bạn. Hãy thử lại sau một chút, biết đâu sẽ có điều thú vị!",
            "🌻 Đừng buồn nhé! Hiện tại trợ lý EasyEdu chưa tìm thấy khóa học nào phù hợp, nhưng hãy thử lại lần nữa để tìm kiếm thêm lựa chọn!",
            "💖 Rất tiếc vì trợ lý EasyEdu chưa tìm thấy khóa học nào cho bạn. Hãy ghé thăm thường xuyên và thử lại lần nữa nhé!",
            "🐾 Mặc dù trợ lý EasyEdu chưa tìm thấy khóa học nào phù hợp, nhưng hãy kiên nhẫn và thử lại sau. Trợ lý EasyEdu luôn ở đây để hỗ trợ bạn!"
        ]
        all_courses = random.choice(no_responses)
        return jsonify({'error_message': {},
                        'code': 1,
                        'data': {
                            'message': all_courses
                        }}), 200


if __name__ == '__main__':
    # Load the data when the application starts
    load_data()
    port = int(os.environ.get('PORT', 5000))
    print("Starting production server...")
    # app.run(debug=True, host='0.0.0.0', port=port)
    serve(app, host='0.0.0.0', port=port)
