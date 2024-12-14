
def merge_data(posts, fact_checks, pairs):
    """
    Merge posts and fact_checks using pairs as the bridge.
    """

    posts = posts.drop_duplicates(subset='post_id')
    fact_checks = fact_checks.drop_duplicates(subset='fact_check_id')
    pairs = pairs.drop_duplicates(subset=['post_id', 'fact_check_id'])


    merged_data = pairs.merge(posts, on='post_id', how='left').merge(fact_checks, on='fact_check_id', how='left')

  
    merged_data.drop(
        columns=['instances_x', 'verdicts', 'ocr_confidence', 'instances_y', 'confidence'], 
        inplace=True, 
        errors='ignore'
    )

    return merged_data
