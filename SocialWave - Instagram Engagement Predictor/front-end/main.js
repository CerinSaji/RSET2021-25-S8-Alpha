document.addEventListener("DOMContentLoaded", () => {
    // Form elements
    const uploadForm = document.getElementById('uploadForm');
    const imageInput = document.getElementById('image');
    const imagePreview = document.getElementById('imagePreview');
    const captionInput = document.getElementById('caption');
    const hashtagsInput = document.getElementById('hashtags');
    const instagramIdInput = document.getElementById('instagramId');
    const validateUsernameButton = document.getElementById('validateUsername');
    const validationResultElement = document.getElementById('validationResult');
    let avgData; //likes and comments
    let userdata; //followers, public account validity
    
    // Tab system
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');
    
    // Caption generation elements
    const generateFromImageButton = document.getElementById('generateFromImage');
    const generateFromPromptButton = document.getElementById('generateFromPrompt');
    const captionPromptInput = document.getElementById('captionPrompt');
    const generateLink = document.getElementById('generateLink');
    const generatedCaptionContainer = document.getElementById('generatedCaptionContainer');
    const generatedCaptionElement = document.getElementById('generatedCaption');
    const useThisCaptionButton = document.getElementById('useThisCaption');
    
    // Hashtag generation elements
    const generatedHashtagsContainer = document.getElementById('generatedHashtagsContainer');
    const generatedHashtagsElement = document.getElementById('generatedHashtags');
    const useTheseHashtagsButton = document.getElementById('useTheseHashtags');
    
    const recommendationsTabs = document.querySelectorAll('[data-rec-tab]');
    
    // Loading and result elements
    const loadingElement = document.getElementById('loading');
    const resultElement = document.getElementById('result');
    const engagementRateElement = document.getElementById('engagementRate');
    const engagementCategoryElement = document.getElementById('engagement-category');
    const feedbackTextElement = document.getElementById('feedbackText');
    const recommendationTextElement = document.getElementById('recommendationText');
    
    const SERVER_URL = 'http://localhost:5000';
    const python_url = 'http://localhost:5001';
    
    // Test server connection on page load
    async function testServerConnection() {
      try {
        const response = await fetch(`${python_url}/test`);
        if (response.ok) {
          console.log('✅ Server connection successful');
        } else {
          const errorText = await response.text();
          console.error(`❌ Server is running but returned an error: ${response.status} - ${errorText}`);
        }
      } catch (error) {
        console.error('❌ Cannot connect to server:', error);
        alert('Cannot connect to the server. Please make sure it is running.');
      }
    }
    
    testServerConnection();
    
    // Polling function for caption status
    async function pollForCaption(jobId, maxAttempts = 60, interval = 1000) {
      console.log(`📊 Starting polling for job ${jobId}`);
      let attempts = 0;
      
      const poll = async () => {
        try {
          if (attempts >= maxAttempts) {
            loadingElement.style.display = 'none';
            captionInput.value = "Caption generation timed out. Please try again.";
            return;
          }
          
          attempts++;
          console.log(`📊 Polling attempt ${attempts} for job ${jobId}`);
          
          const response = await fetch(`${python_url}/get-caption-status/${jobId}`);
          const data = await response.json();
          
          if (data.status === 'completed') {
            loadingElement.style.display = 'none';
            captionInput.value = data.caption;
            console.log('✅ Caption received:', data.caption);
            return;
          } else if (data.status === 'error') {
            loadingElement.style.display = 'none';
            captionInput.value = "Error generating caption.";
            console.error('❌ Caption generation error:', data.message);
            return;
          } else {
            setTimeout(poll, interval);
          }
        } catch (error) {
          loadingElement.style.display = 'none';
          captionInput.value = "Error checking caption status.";
          console.error('❌ Polling error:', error);
        }
      };
      
      await poll();
    }
    
    function splitCaptionAndHashtags(data) {
      // Match all hashtags using RegEx
      const hashtags = data.match(/#\w+/g) || [];
    
      // Remove hashtags from the original string
      const caption = data.replace(/#\w+/g, "").trim();
    
      // Join hashtags into one string
      const hashtagsStr = hashtags.join(" ");
    
      return { caption, hashtags: hashtagsStr };
    }
    
    // Tab system functionality
    tabButtons.forEach(button => {
      button.addEventListener('click', () => {
        // Remove active class from all buttons and panes
        tabButtons.forEach(btn => btn.classList.remove('active'));
        tabPanes.forEach(pane => pane.classList.remove('active'));
        
        // Add active class to current button
        button.classList.add('active');
        
        // Show corresponding tab pane
        const tabId = button.getAttribute('data-tab');
        if (tabId === 'write') {
          document.getElementById('writeTab').classList.add('active');
        } else if (tabId === 'generate') {
          document.getElementById('generateTab').classList.add('active');
        }
      });
    });
  
    // Recommendation tabs functionality
    recommendationsTabs.forEach(button => {
      button.addEventListener('click', () => {
        // Remove active class from all buttons and panes
        recommendationsTabs.forEach(btn => btn.classList.remove('active'));
        document.querySelectorAll('.recommendation-tab-pane').forEach(pane => pane.classList.remove('active'));
        
        // Add active class to current button
        button.classList.add('active');
        
        // Show corresponding tab pane
        const tabId = button.getAttribute('data-rec-tab');
        if (tabId === 'hashtags') {
          document.getElementById('hashtagsTab').classList.add('active');
        } else if (tabId === 'captions') {
          document.getElementById('captionsTab').classList.add('active');
        }
      });
    });
    
    // Preview image functionality
    imageInput.addEventListener('change', function() {
      const file = this.files[0];
      if (file) {
        // Validate file type
        if (!file.type.match('image.*')) {
          alert('Please select an image file (JPEG, PNG, etc.)');
          this.value = ''; // Clear the input
          imagePreview.innerHTML = '';
          return;
        }
        
        // Validate file size (5MB max)
        if (file.size > 5 * 1024 * 1024) {
          alert('Image is too large. Please select an image under 5MB');
          this.value = ''; // Clear the input
          imagePreview.innerHTML = '';
          return;
        }
        
        const reader = new FileReader();
        reader.onload = function(e) {
          imagePreview.innerHTML = `<img src="${e.target.result}" alt="Image Preview" class="preview-img">`;
        };
        reader.readAsDataURL(file);
      } else {
        imagePreview.innerHTML = '';
      }
    });
    
    // Handle the "Generate from Prompt" button
    generateFromPromptButton.addEventListener('click', function() {
      captionPromptInput.classList.remove('hidden');
      generateLink.classList.remove('hidden');
      captionPromptInput.focus();
    });
    
    // Handle the "Generate!" link click
    generateLink.addEventListener('click', async function(event) {
      event.preventDefault();
      const prompt = captionPromptInput.value.trim();
      
      if (!prompt) {
        alert('Please enter a prompt for your caption and hashtags.');
        return;
      }
      
      try {      
        console.log('📤 Sending prompt to server:', prompt);
        const response = await fetch(`${python_url}/generate-caption-from-prompt`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ prompt })
        });
        
        const responseText = await response.text();
        console.log('📥 Raw response:', responseText);
        
        // Parse as JSON
        let data;
        try {
          data = JSON.parse(responseText);
        } catch (parseError) {
          console.error('❌ Invalid JSON response:', responseText);
          alert('The server returned an invalid response. Check console for details.');
          loadingElement.style.display = 'none';
          return;
        }
              
        if (!response.ok) {
          console.error('❌ Server Error:', data);
          alert(`Error generating caption: ${data.error || 'Unknown error'}`);
          return;
        }
        
        console.log('📥 Caption received:', data);
  
        const { caption, hashtags } = splitCaptionAndHashtags(data.caption);
  
        console.log("Caption:", caption);
        console.log("Hashtags:", hashtags);
        
        // Display the generated caption
        generatedCaptionElement.textContent = caption;
        generatedCaptionContainer.classList.remove('hidden');
        
        generatedHashtagsElement.textContent = hashtags;
        generatedHashtagsContainer.classList.remove('hidden');
      } catch (error) {
        loadingElement.style.display = 'none';
        
        console.error('❌ Fetch Error:', error);
        alert('An error occurred while generating the caption. Please check if the server is running.');
      }
    });
    
    // Handle the "Generate from Image" button
    generateFromImageButton.addEventListener('click', async function() {
      const file = imageInput.files[0];
      if (!file) {
        alert('Please select an image first.');
        return;
      }
      
      //loadingElement.style.display = 'block';
      
      const formData = new FormData();
      formData.append('image', file);
      
      try {
        console.log('📤 Sending image to server');
        const response = await fetch(`${python_url}/generate-caption-from-image`, {
          method: 'POST',
          body: formData
        });
        
        const responseText = await response.text();
        console.log('📥 Raw response:', responseText);
        
        let data;
        try {
          data = JSON.parse(responseText);
        } catch (parseError) {
          console.error('❌ Invalid JSON response:', responseText);
          alert('The server returned an invalid response. Check console for details.');
          loadingElement.style.display = 'none';
          return;
        }
        
        //loadingElement.style.display = 'none';
        
        if (!response.ok) {
          console.error('❌ Server Error:', data);
          alert(`Error generating caption: ${data.error || 'Unknown error'}`);
          return;
        }
        
        console.log('✅ Caption received:', data);
        
        const { caption, hashtags } = splitCaptionAndHashtags(data.caption);
  
        console.log("Caption:", caption);
        console.log("Hashtags:", hashtags);
        
        // Display the generated caption
        generatedCaptionElement.textContent = caption;
        generatedCaptionContainer.classList.remove('hidden');
        
        generatedHashtagsElement.textContent = hashtags;
        generatedHashtagsContainer.classList.remove('hidden');
      } catch (error) {
        //loadingElement.style.display = 'none';
        console.error('❌ Fetch Error:', error);
        alert('An error occurred while generating the caption. Please check if the server is running.');
      }
    });
    
    // Use generated caption button
    useThisCaptionButton.addEventListener('click', function() {
      captionInput.value = generatedCaptionElement.textContent;
      // Switch to Write tab
      document.querySelector('[data-tab="write"]').click();
    });
    
    // Use generated hashtags button
    useTheseHashtagsButton.addEventListener('click', function() {
      hashtagsInput.value = generatedHashtagsElement.textContent;
      // Switch to Write tab
      document.querySelector('[data-tab="write"]').click();
    });
  
    let validatingInterval = null;
  
    function startValidatingAnimation(element) {
      const baseText = " Validating";
      let dotCount = 0;
    
      validatingInterval = setInterval(() => {
        dotCount = (dotCount + 1) % 4;
        const dots = ".".repeat(dotCount);
        element.textContent = baseText + dots;
      }, 400); // Adjust speed here if needed
    }
    
    function stopValidatingAnimation() {
      clearInterval(validatingInterval);
      validatingInterval = null;
    }
    
    
    validateUsernameButton.addEventListener('click', async () => {
      const username = instagramIdInput.value.trim();
    
      if (!username) {
        validationResultElement.textContent = 'Please enter an Instagram username to validate.';
        validationResultElement.style.color = 'black';
        return;
      }
    
      const cleanUsername = username.startsWith('@') ? username.slice(1) : username;
    
      validationResultElement.textContent = '';
      validationResultElement.style.color = 'black';
    
      startValidatingAnimation(validationResultElement);
    
      try {
        console.log(`📤 Validating Instagram username: ${cleanUsername}`);
    
        const response = await fetch(`${python_url}/validate-instagram-username`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username: cleanUsername }),
        });
    
        userdata = await response.json();
        stopValidatingAnimation();
    
        if (userdata.valid && userdata.exists && userdata.is_public) {
          validationResultElement.textContent = `✓ @${cleanUsername} is a valid public Instagram account with ${userdata.followers} followers`;
    
          // Show calculating message
          const calcMessage = document.createElement('div');
          calcMessage.textContent = 'Calculating average likes and comments...';
          calcMessage.style.color = 'black';
          validationResultElement.appendChild(calcMessage);
    
          // Fetch average engagement separately
          const avgResponse = await fetch(`${python_url}/instagram-average-engagement`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ username: cleanUsername }),
          });
    
          avgData = await avgResponse.json();
    
          // Replace calculating message with results
          calcMessage.textContent = `📊 Average likes: ${avgData.average_likes}, comments: ${avgData.average_comments}`;
    
        } else if (data.valid && data.exists && !data.is_public) {
          validationResultElement.textContent = `🔒 @${cleanUsername} is a private Instagram account`;
          instagramIdInput.value = '';
          animatePlaceholder(instagramIdInput, "Please provide a public account's username.");
          validationResultElement.style.color = 'black';
    
        } else if (!data.exists) {
          validationResultElement.textContent = `❌ @${cleanUsername} does not exist`;
          instagramIdInput.value = '';
          animatePlaceholder(instagramIdInput, "Please provide an existing account's username.");
          validationResultElement.style.color = 'black';
    
        } else {
          validationResultElement.textContent = '⚠️ Something went wrong. Please try again.';
          validationResultElement.style.color = 'black';
        }
    
      } catch (error) {
        stopValidatingAnimation();
        console.error('❌ Validation Error:', error.message);
        validationResultElement.textContent = 'Error validating username. Please try again.';
        validationResultElement.style.color = 'black';
      }
    });    
  
    function animatePlaceholder(inputElement, newText, speed = 50) {
      let i = 0;
    
      const interval = setInterval(() => {
        inputElement.placeholder = newText.slice(0, i);
        i++;
    
        if (i > newText.length) {
          clearInterval(interval);
        }
      }, speed);
    }
  
    // Handle form submission
    uploadForm.addEventListener('submit', async function(event) {
      event.preventDefault();
      
      const username = instagramIdInput.value.trim();
      const caption = captionInput.value.trim();
      const hashtags = hashtagsInput.value.trim();
      const file = imageInput.files[0];
      
      if (!username || !file) {
        alert('Please fill in all required fields (username and image).');
        return;
      }
      
      loadingElement.style.display = 'block';
      resultElement.style.display = 'none';
      
      try {
  
        const payload = {
          average_likes: avgData.average_likes,
          average_comments: avgData.average_comments,
          followers: userdata.followers,
          caption,
          hashtags
        };
  
        console.log('📤 Sending data for engagement prediction');
        
        const response = await fetch(`${python_url}/predict-engagement`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        
        const data = await response.json();
        
        loadingElement.style.display = 'none';
        
        if (!response.ok) {
          console.error('❌ Server Error:', data);
          alert(`Error predicting engagement: ${data.error || 'Unknown error'}`);
          return;
        }
        
        console.log('✅ Prediction received:', data);
        
        // Display the result
        engagementRateElement.textContent = data.adjusted_engagement_rate.toFixed(1);
        engagementCategoryElement.textContent = 'Engagement Category: ' + data.engagement_category;
        
        // Set feedback text based on engagement rate
        let feedbackText = '';
        let recommendationText = '';
        
        if (data.adjusted_engagement_rate < 20) {
          feedbackText = 'Very low engagement predicted. This post might not perform well.';
          recommendationText = 'Try using more relevant hashtags and a more engaging caption.';
        } else if (data.adjusted_engagement_rate < 40) {
          feedbackText = 'Average engagement predicted. This is typical for most posts.';
          recommendationText = 'For better results, try posting when your audience is most active.';
        } else if (data.adjusted_engagement_rate <= 60) {
          feedbackText = 'Good engagement predicted! This post should perform well.';
          recommendationText = 'Make sure to respond to comments to boost engagement further.';
        } else {
          feedbackText = 'Excellent engagement predicted! This post has high viral potential.';
          recommendationText = 'Consider promoting this post to reach an even wider audience.';
        }
        
        // Add analysis from the server if available
        if (data.analysis) {
          feedbackText += ` ${data.analysis.feedback || ''}`;
          recommendationText += ` ${data.analysis.recommendation || ''}`;
        }
        
        feedbackTextElement.textContent = feedbackText;
        recommendationTextElement.textContent = recommendationText;
        
        resultElement.style.display = 'block';
        
      } catch (error) {
        loadingElement.style.display = 'none';
        console.error('❌ Fetch Error:', error);
        alert('An error occurred while predicting engagement. Please check if the server is running.');
      }
    });
  
    // Get DOM elements
    const sidebar = document.getElementById('sidebar');
    const contentArea = document.getElementById('contentArea');
    const toggleBtn = document.getElementById('toggleSidebar');
    const toggleIcon = document.getElementById('toggleIcon');
    const navItems = document.querySelectorAll('.sidebar-nav-item');
    const pages = document.querySelectorAll('.page-content');
    
    // Set username if available
    const storedUsername = localStorage.getItem('username') || 'User';
    document.getElementById('username').textContent = storedUsername;
    
    // Toggle sidebar function
    function toggleSidebar() {
      sidebar.classList.toggle('expanded');
      if (sidebar.classList.contains('expanded')) {
        toggleIcon.textContent = '❯';
      } else {
        toggleIcon.textContent = '❮';
      }
    }
    
    // Toggle sidebar when button is clicked
    toggleBtn.addEventListener('click', function(e) {
      e.stopPropagation();
      toggleSidebar();
    });
    
    // Handle navigation menu clicks
    navItems.forEach(item => {
      item.addEventListener('click', function() {
        // Remove active class from all nav items
        navItems.forEach(nav => nav.classList.remove('active'));
        
        // Add active class to clicked item
        this.classList.add('active');
        
        // Hide all pages
        pages.forEach(page => page.classList.add('hidden'));
        
        // Show the selected page
        const pageId = this.getAttribute('data-page') + 'Page';
        document.getElementById(pageId).classList.remove('hidden');
        
        if (this.getAttribute('data-page') === 'saved') {
            loadSavedPosts();
          }
          
        // On mobile, close sidebar after navigation
        if (window.innerWidth <= 768) {
          sidebar.classList.remove('expanded');
          toggleIcon.textContent = '❮';
        }
      });
    });
    
    // Handle logout button
    document.getElementById('logoutBtn').addEventListener('click', function() {
      if (confirm('Are you sure you want to logout?')) {
        localStorage.removeItem('username');
        localStorage.removeItem('userToken');
        alert('You have been logged out successfully.');
        // window.location.href = 'login.html'; // Uncomment when you have a login page
      }
    });
    
    // Close sidebar when clicking outside (for mobile)
    document.addEventListener('click', function(e) {
      if (window.innerWidth <= 768 && 
          sidebar.classList.contains('expanded') && 
          !sidebar.contains(e.target) && 
          e.target !== toggleBtn) {
        sidebar.classList.remove('expanded');
        toggleIcon.textContent = '❮';
      }
    });
  
    const saveAnalysisBtn = document.getElementById('saveAnalysisBtn');
const saveStatus = document.getElementById('saveStatus');

saveAnalysisBtn.addEventListener('click', async function () {
  const email = localStorage.getItem('userEmail');
  const imageFile = document.getElementById('image').files[0];
  const caption = document.getElementById('caption').value;
  const hashtags = document.getElementById('hashtags').value;
  const engagementRate = document.getElementById('engagementRate').textContent;

  if (!email || !imageFile || !caption || !hashtags || !engagementRate) {
    saveStatus.textContent = 'Please fill in all fields before saving.';
    saveStatus.style.color = 'red';
    return;
  }

  const formData = new FormData();
  formData.append('email', email);
  formData.append('image', imageFile);
  formData.append('caption', caption);
  formData.append('hashtags', hashtags);
  formData.append('engagementRate', parseFloat(engagementRate));

  try {
    const response = await fetch('http://localhost:5000/upload', {
      method: 'POST',
      body: formData,
    });

    const data = await response.json();
    saveStatus.textContent = data.message;
    saveStatus.style.color = 'green';

    // Optional: clear the form fields or disable button
  } catch (error) {
    console.error('Error saving post:', error);
    saveStatus.textContent = 'Failed to save analysis. Try again later.';
    saveStatus.style.color = 'red';
  }

  // Clear status after 3 seconds
  setTimeout(() => {
    saveStatus.textContent = '';
  }, 3000);
});
// ===== Load Saved Posts Logic =====
async function loadSavedPosts() {
    const email = localStorage.getItem('userEmail');
    const savedPostsContainer = document.getElementById('savedPostsContainer');
  
    if (!email) {
      savedPostsContainer.innerHTML = '<p>Please log in to view your saved posts.</p>';
      return;
    }
  
    try {
      const response = await fetch(`http://localhost:5000/posts/${email}`);
      const data = await response.json();
  
      savedPostsContainer.innerHTML = ''; // Clear previous content
  
      if (!data.posts || data.posts.length === 0) {
        savedPostsContainer.innerHTML = '<p>No saved posts yet.</p>';
        return;
      }
  
      data.posts.forEach(post => {
        const card = document.createElement('div');
        card.classList.add('saved-post-card');
  
        const base64Image = post.image
          ? `data:${post.image.contentType};base64,${post.image.data}`
          : null;
  
        card.innerHTML = `
          ${base64Image ? `<img src="${base64Image}" class="saved-post-image" />` : ''}
          <div class="saved-post-details">
            <p><strong>Caption:</strong> ${post.caption}</p>
            <p><strong>Hashtags:</strong> ${post.hashtags}</p>
            <p><strong>Engagement Rate:</strong> ${post.engagementRate || 'N/A'}%</p>
            <p><em>Saved on: ${new Date(post.createdAt).toLocaleString()}</em></p>
          </div>
        `;
  
        savedPostsContainer.appendChild(card);
      });
    } catch (err) {
      console.error('Error loading saved posts:', err);
      savedPostsContainer.innerHTML = '<p>Error fetching posts. Try again later.</p>';
    }
}
  });