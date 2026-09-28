#define GLM_FORCE_RADIANS
#define GLM_FORCE_DEPTH_ZERO_TO_ONE

#include <GLFW/glfw3.h>
#include <vulkan/vulkan.hpp>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "geometry.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
constexpr uint32_t kRenderWidth = 854;
constexpr uint32_t kRenderHeight = 480;
constexpr uint32_t kOutputWidth = 1280;
constexpr uint32_t kOutputHeight = 720;
#ifdef SCENE_TOILET
constexpr const char* kWindowTitleCheckerboard = "Vulkan Checkerboard Toilet (480p -> 720p)";
constexpr const char* kWindowTitleBilinear = "Vulkan Bilinear Upscale Toilet (480p -> 720p)";
constexpr const char* kSceneName = "toilet";
#else
constexpr const char* kWindowTitleCheckerboard = "Vulkan Checkerboard Cube (480p -> 720p)";
constexpr const char* kWindowTitleBilinear = "Vulkan Bilinear Upscale (480p -> 720p)";
constexpr const char* kSceneName = "cube";
#endif
constexpr vk::Format kOffscreenColorFormat = vk::Format::eR8G8B8A8Unorm;
constexpr vk::Format kDepthFormat = vk::Format::eD32Sfloat;

struct CubeUBO
{
    glm::mat4 mvp;
    glm::mat4 model;
    glm::vec4 lightDir;
};

struct ReconstructUBO
{
    glm::mat4 invViewProj;
    glm::mat4 prevViewProj;
    glm::mat4 objectReproject;
    glm::vec2 renderSize;
    glm::vec2 outputSize;
    uint32_t frameIndex;
    uint32_t firstFrame;
    uint32_t mode;
    uint32_t pad;
};

struct QueueFamilyIndices
{
    std::optional<uint32_t> graphics;
    std::optional<uint32_t> present;

    bool complete() const
    {
        return graphics.has_value() && present.has_value();
    }
};

struct SwapchainSupport
{
    vk::SurfaceCapabilitiesKHR capabilities;
    std::vector<vk::SurfaceFormatKHR> formats;
    std::vector<vk::PresentModeKHR> presentModes;
};

struct ImageResource
{
    vk::Image image;
    vk::DeviceMemory memory;
    vk::ImageView view;
};

struct BufferResource
{
    vk::Buffer buffer;
    vk::DeviceMemory memory;
};

struct OffscreenTarget
{
    ImageResource color;
    ImageResource depth;
};

std::vector<uint32_t> readSpirv(const std::filesystem::path& path)
{
    std::ifstream file(path, std::ios::ate | std::ios::binary);
    if (!file)
    {
        throw std::runtime_error("failed to open shader: " + path.string());
    }

    const size_t size = static_cast<size_t>(file.tellg());
    if (size == 0 || (size % 4) != 0)
    {
        throw std::runtime_error("shader is empty or not aligned: " + path.string());
    }

    std::vector<uint32_t> buffer(size / 4);
    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), static_cast<std::streamsize>(size));
    return buffer;
}

VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                             VkDebugUtilsMessageTypeFlagsEXT,
                                             const VkDebugUtilsMessengerCallbackDataEXT* data,
                                             void*)
{
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
    {
        std::cerr << "validation: " << data->pMessage << '\n';
    }
    return VK_FALSE;
}
}

class CheckerboardApp
{
public:
    void run()
    {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:
    GLFWwindow* window = nullptr;
    bool framebufferResized = false;
    bool checkerboardEnabled = true;
    bool spaceWasDown = false;

    vk::Instance instance;
    vk::DebugUtilsMessengerEXT debugMessenger;
    PFN_vkCreateDebugUtilsMessengerEXT pfnCreateDebugMessenger = nullptr;
    PFN_vkDestroyDebugUtilsMessengerEXT pfnDestroyDebugMessenger = nullptr;
    vk::SurfaceKHR surface;
    vk::PhysicalDevice physicalDevice;
    vk::Device device;
    vk::Queue graphicsQueue;
    vk::Queue presentQueue;
    uint32_t graphicsFamily = 0;
    uint32_t presentFamily = 0;

    vk::SwapchainKHR swapchain;
    vk::Format swapchainFormat = vk::Format::eB8G8R8A8Srgb;
    vk::Extent2D swapchainExtent;
    std::vector<vk::Image> swapchainImages;
    std::vector<vk::ImageView> swapchainViews;

    vk::DescriptorSetLayout cubeSetLayout;
    vk::DescriptorSetLayout reconstructSetLayout;
    vk::PipelineLayout cubePipelineLayout;
    vk::PipelineLayout reconstructPipelineLayout;
    vk::Pipeline cubePipeline;
    vk::Pipeline reconstructPipeline;

    OffscreenTarget offscreen[2];
    vk::Sampler linearSampler;
    vk::Sampler nearestSampler;

    BufferResource vertexBuffer;
    BufferResource indexBuffer;
    uint32_t indexCount = 0;
    BufferResource cubeUboBuffer;
    BufferResource reconstructUboBuffer;
    void* cubeUboMapped = nullptr;
    void* reconstructUboMapped = nullptr;

    vk::DescriptorPool descriptorPool;
    vk::DescriptorSet cubeDescriptorSet;
    vk::DescriptorSet reconstructDescriptorSets[2];

    vk::CommandPool commandPool;
    vk::CommandBuffer commandBuffer;
    vk::Semaphore imageAvailable;
    std::vector<vk::Semaphore> renderFinished;
    vk::Fence inFlight;

    glm::mat4 prevViewProj{1.0f};
    glm::mat4 prevModel{1.0f};
    uint32_t frameIndex = 0;
    bool firstFrame = true;
    uint32_t currentTarget = 0;

    static void onFramebufferResize(GLFWwindow* glfwWindow, int, int)
    {
        auto* app = static_cast<CheckerboardApp*>(glfwGetWindowUserPointer(glfwWindow));
        app->framebufferResized = true;
    }

    void initWindow()
    {
        if (!glfwInit())
        {
            throw std::runtime_error("failed to initialize GLFW");
        }
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
        window = glfwCreateWindow(kOutputWidth, kOutputHeight, kWindowTitleCheckerboard, nullptr, nullptr);
        if (!window)
        {
            throw std::runtime_error("failed to create window");
        }
        glfwSetWindowUserPointer(window, this);
        glfwSetFramebufferSizeCallback(window, onFramebufferResize);
        std::cout << "Rendering a " << kSceneName << " at 480p and reconstructing to 720p with checkerboard upscaling.\n"
                  << "SPACE toggles naive bilinear upscale. ESC quits.\n";
    }

    void initVulkan()
    {
        createInstance();
        setupDebugMessenger();
        createSurface();
        pickPhysicalDevice();
        createLogicalDevice();
        createSwapchain();
        createOffscreenTargets();
        createSamplers();
        createDescriptorSetLayouts();
        createPipelines();
        createMeshBuffers();
        createUniformBuffers();
        createDescriptorPoolAndSets();
        createCommandObjects();
        createSyncObjects();
    }

    void createInstance()
    {
        uint32_t glfwCount = 0;
        const char** glfwExt = glfwGetRequiredInstanceExtensions(&glfwCount);
        std::vector<const char*> extensions(glfwExt, glfwExt + glfwCount);
#ifndef NDEBUG
        extensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        const std::vector<const char*> layers = {"VK_LAYER_KHRONOS_validation"};
#endif

        vk::ApplicationInfo appInfo{};
        appInfo.pApplicationName = "vulkan-cube-checkerboarding";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "none";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_3;

        vk::InstanceCreateInfo createInfo{};
        createInfo.pApplicationInfo = &appInfo;
        createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
        createInfo.ppEnabledExtensionNames = extensions.data();
#ifndef NDEBUG
        if (hasValidationLayer())
        {
            createInfo.enabledLayerCount = static_cast<uint32_t>(layers.size());
            createInfo.ppEnabledLayerNames = layers.data();
        }
#endif

        instance = vk::createInstance(createInfo);
    }

#ifndef NDEBUG
    bool hasValidationLayer() const
    {
        const auto layers = vk::enumerateInstanceLayerProperties();
        return std::any_of(layers.begin(), layers.end(), [](const vk::LayerProperties& layer) {
            return std::strcmp(layer.layerName, "VK_LAYER_KHRONOS_validation") == 0;
        });
    }
#endif

    void setupDebugMessenger()
    {
#ifdef NDEBUG
        return;
#endif
        pfnCreateDebugMessenger = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            instance.getProcAddr("vkCreateDebugUtilsMessengerEXT"));
        pfnDestroyDebugMessenger = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            instance.getProcAddr("vkDestroyDebugUtilsMessengerEXT"));
        if (!pfnCreateDebugMessenger)
        {
            return;
        }

        VkDebugUtilsMessengerCreateInfoEXT info{};
        info.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
        info.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                               VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        info.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                           VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        info.pfnUserCallback = debugCallback;
        VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
        pfnCreateDebugMessenger(instance, &info, nullptr, &messenger);
        debugMessenger = messenger;
    }

    void createSurface()
    {
        VkSurfaceKHR raw = VK_NULL_HANDLE;
        if (glfwCreateWindowSurface(instance, window, nullptr, &raw) != VK_SUCCESS)
        {
            throw std::runtime_error("failed to create window surface");
        }
        surface = raw;
    }

    QueueFamilyIndices findQueueFamilies(vk::PhysicalDevice gpu) const
    {
        QueueFamilyIndices indices;
        const auto families = gpu.getQueueFamilyProperties();
        for (uint32_t i = 0; i < families.size(); ++i)
        {
            if (families[i].queueFlags & vk::QueueFlagBits::eGraphics)
            {
                indices.graphics = i;
            }
            if (gpu.getSurfaceSupportKHR(i, surface))
            {
                indices.present = i;
            }
            if (indices.complete())
            {
                break;
            }
        }
        return indices;
    }

    SwapchainSupport querySwapchainSupport(vk::PhysicalDevice gpu) const
    {
        return {
            gpu.getSurfaceCapabilitiesKHR(surface),
            gpu.getSurfaceFormatsKHR(surface),
            gpu.getSurfacePresentModesKHR(surface),
        };
    }

    bool isDeviceSuitable(vk::PhysicalDevice gpu) const
    {
        const auto indices = findQueueFamilies(gpu);
        const auto extensions = gpu.enumerateDeviceExtensionProperties();
        const bool hasSwapchain = std::any_of(extensions.begin(), extensions.end(), [](const vk::ExtensionProperties& ext) {
            return std::strcmp(ext.extensionName, VK_KHR_SWAPCHAIN_EXTENSION_NAME) == 0;
        });
        if (!indices.complete() || !hasSwapchain)
        {
            return false;
        }
        const auto support = querySwapchainSupport(gpu);
        return !support.formats.empty() && !support.presentModes.empty();
    }

    void pickPhysicalDevice()
    {
        const auto devices = instance.enumeratePhysicalDevices();
        for (auto gpu : devices)
        {
            if (isDeviceSuitable(gpu))
            {
                physicalDevice = gpu;
                break;
            }
        }
        if (!physicalDevice)
        {
            throw std::runtime_error("no suitable Vulkan device");
        }
        const auto indices = findQueueFamilies(physicalDevice);
        graphicsFamily = *indices.graphics;
        presentFamily = *indices.present;
    }

    void createLogicalDevice()
    {
        std::vector<vk::DeviceQueueCreateInfo> queueInfos;
        const float priority = 1.0f;
        const std::vector<uint32_t> uniqueFamilies = (graphicsFamily == presentFamily)
                                                         ? std::vector<uint32_t>{graphicsFamily}
                                                         : std::vector<uint32_t>{graphicsFamily, presentFamily};
        for (uint32_t family : uniqueFamilies)
        {
            vk::DeviceQueueCreateInfo info{};
            info.queueFamilyIndex = family;
            info.queueCount = 1;
            info.pQueuePriorities = &priority;
            queueInfos.push_back(info);
        }

        vk::PhysicalDeviceFeatures features{};
        vk::PhysicalDeviceVulkan13Features features13{};
        features13.dynamicRendering = VK_TRUE;
        features13.synchronization2 = VK_TRUE;

        vk::PhysicalDeviceVulkan11Features features11{};
        features11.pNext = &features13;
        features11.shaderDrawParameters = VK_TRUE;

        const char* deviceExtensions[] = {VK_KHR_SWAPCHAIN_EXTENSION_NAME};
        vk::DeviceCreateInfo createInfo{};
        createInfo.pNext = &features11;
        createInfo.queueCreateInfoCount = static_cast<uint32_t>(queueInfos.size());
        createInfo.pQueueCreateInfos = queueInfos.data();
        createInfo.enabledExtensionCount = 1;
        createInfo.ppEnabledExtensionNames = deviceExtensions;
        createInfo.pEnabledFeatures = &features;

        device = physicalDevice.createDevice(createInfo);
        graphicsQueue = device.getQueue(graphicsFamily, 0);
        presentQueue = device.getQueue(presentFamily, 0);
    }

    vk::SurfaceFormatKHR chooseSurfaceFormat(const std::vector<vk::SurfaceFormatKHR>& formats) const
    {
        for (const auto& format : formats)
        {
            if (format.format == vk::Format::eB8G8R8A8Srgb &&
                format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
            {
                return format;
            }
        }
        return formats.front();
    }

    vk::PresentModeKHR choosePresentMode(const std::vector<vk::PresentModeKHR>& modes) const
    {
        for (auto mode : modes)
        {
            if (mode == vk::PresentModeKHR::eMailbox)
            {
                return mode;
            }
        }
        return vk::PresentModeKHR::eFifo;
    }

    vk::Extent2D chooseExtent(const vk::SurfaceCapabilitiesKHR& capabilities) const
    {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
        {
            return capabilities.currentExtent;
        }
        int width = 0;
        int height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        vk::Extent2D extent{
            std::clamp(static_cast<uint32_t>(width), capabilities.minImageExtent.width, capabilities.maxImageExtent.width),
            std::clamp(static_cast<uint32_t>(height), capabilities.minImageExtent.height, capabilities.maxImageExtent.height),
        };
        return extent;
    }

    void createSwapchain()
    {
        const auto support = querySwapchainSupport(physicalDevice);
        const auto surfaceFormat = chooseSurfaceFormat(support.formats);
        const auto presentMode = choosePresentMode(support.presentModes);
        swapchainExtent = chooseExtent(support.capabilities);
        swapchainFormat = surfaceFormat.format;

        uint32_t imageCount = support.capabilities.minImageCount + 1;
        if (support.capabilities.maxImageCount > 0 && imageCount > support.capabilities.maxImageCount)
        {
            imageCount = support.capabilities.maxImageCount;
        }

        vk::SwapchainCreateInfoKHR info{};
        info.surface = surface;
        info.minImageCount = imageCount;
        info.imageFormat = swapchainFormat;
        info.imageColorSpace = surfaceFormat.colorSpace;
        info.imageExtent = swapchainExtent;
        info.imageArrayLayers = 1;
        info.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;
        const uint32_t families[] = {graphicsFamily, presentFamily};
        if (graphicsFamily != presentFamily)
        {
            info.imageSharingMode = vk::SharingMode::eConcurrent;
            info.queueFamilyIndexCount = 2;
            info.pQueueFamilyIndices = families;
        }
        else
        {
            info.imageSharingMode = vk::SharingMode::eExclusive;
        }
        info.preTransform = support.capabilities.currentTransform;
        info.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
        info.presentMode = presentMode;
        info.clipped = VK_TRUE;

        swapchain = device.createSwapchainKHR(info);
        swapchainImages = device.getSwapchainImagesKHR(swapchain);
        swapchainViews.resize(swapchainImages.size());
        for (size_t i = 0; i < swapchainImages.size(); ++i)
        {
            swapchainViews[i] = createImageView(swapchainImages[i], swapchainFormat, vk::ImageAspectFlagBits::eColor);
        }
    }

    vk::ImageView createImageView(vk::Image image, vk::Format format, vk::ImageAspectFlags aspect) const
    {
        vk::ImageViewCreateInfo info{};
        info.image = image;
        info.viewType = vk::ImageViewType::e2D;
        info.format = format;
        info.subresourceRange.aspectMask = aspect;
        info.subresourceRange.levelCount = 1;
        info.subresourceRange.layerCount = 1;
        return device.createImageView(info);
    }

    uint32_t findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) const
    {
        const auto memory = physicalDevice.getMemoryProperties();
        for (uint32_t i = 0; i < memory.memoryTypeCount; ++i)
        {
            if ((typeFilter & (1u << i)) && (memory.memoryTypes[i].propertyFlags & properties) == properties)
            {
                return i;
            }
        }
        throw std::runtime_error("no compatible memory type");
    }

    ImageResource createImage(uint32_t width, uint32_t height, vk::Format format, vk::ImageUsageFlags usage,
                              vk::ImageAspectFlags aspect)
    {
        vk::ImageCreateInfo info{};
        info.imageType = vk::ImageType::e2D;
        info.extent = vk::Extent3D{width, height, 1};
        info.mipLevels = 1;
        info.arrayLayers = 1;
        info.format = format;
        info.tiling = vk::ImageTiling::eOptimal;
        info.initialLayout = vk::ImageLayout::eUndefined;
        info.usage = usage;
        info.samples = vk::SampleCountFlagBits::e1;
        info.sharingMode = vk::SharingMode::eExclusive;

        ImageResource resource;
        resource.image = device.createImage(info);
        const auto requirements = device.getImageMemoryRequirements(resource.image);
        vk::MemoryAllocateInfo alloc{};
        alloc.allocationSize = requirements.size;
        alloc.memoryTypeIndex = findMemoryType(requirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal);
        resource.memory = device.allocateMemory(alloc);
        device.bindImageMemory(resource.image, resource.memory, 0);
        resource.view = createImageView(resource.image, format, aspect);
        return resource;
    }

    void destroyImage(ImageResource& resource)
    {
        if (resource.view)
        {
            device.destroyImageView(resource.view);
        }
        if (resource.image)
        {
            device.destroyImage(resource.image);
        }
        if (resource.memory)
        {
            device.freeMemory(resource.memory);
        }
        resource = {};
    }

    void createOffscreenTargets()
    {
        const vk::ImageUsageFlags colorUsage =
            vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled;
        const vk::ImageUsageFlags depthUsage =
            vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled;
        for (auto& target : offscreen)
        {
            target.color = createImage(kRenderWidth, kRenderHeight, kOffscreenColorFormat, colorUsage,
                                       vk::ImageAspectFlagBits::eColor);
            target.depth = createImage(kRenderWidth, kRenderHeight, kDepthFormat, depthUsage,
                                       vk::ImageAspectFlagBits::eDepth);
        }
    }

    void createSamplers()
    {
        vk::SamplerCreateInfo info{};
        info.magFilter = vk::Filter::eLinear;
        info.minFilter = vk::Filter::eLinear;
        info.mipmapMode = vk::SamplerMipmapMode::eNearest;
        info.addressModeU = vk::SamplerAddressMode::eClampToEdge;
        info.addressModeV = vk::SamplerAddressMode::eClampToEdge;
        info.addressModeW = vk::SamplerAddressMode::eClampToEdge;
        linearSampler = device.createSampler(info);

        info.magFilter = vk::Filter::eNearest;
        info.minFilter = vk::Filter::eNearest;
        nearestSampler = device.createSampler(info);
    }

    void createDescriptorSetLayouts()
    {
        vk::DescriptorSetLayoutBinding cubeBinding{};
        cubeBinding.binding = 0;
        cubeBinding.descriptorType = vk::DescriptorType::eUniformBuffer;
        cubeBinding.descriptorCount = 1;
        cubeBinding.stageFlags = vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment;
        vk::DescriptorSetLayoutCreateInfo cubeInfo{};
        cubeInfo.bindingCount = 1;
        cubeInfo.pBindings = &cubeBinding;
        cubeSetLayout = device.createDescriptorSetLayout(cubeInfo);

        std::array<vk::DescriptorSetLayoutBinding, 7> reconstructBindings{};
        for (uint32_t i = 0; i < 4; ++i)
        {
            reconstructBindings[i].binding = i;
            reconstructBindings[i].descriptorType = vk::DescriptorType::eSampledImage;
            reconstructBindings[i].descriptorCount = 1;
            reconstructBindings[i].stageFlags = vk::ShaderStageFlagBits::eFragment;
        }
        reconstructBindings[4].binding = 4;
        reconstructBindings[4].descriptorType = vk::DescriptorType::eSampler;
        reconstructBindings[4].descriptorCount = 1;
        reconstructBindings[4].stageFlags = vk::ShaderStageFlagBits::eFragment;
        reconstructBindings[5].binding = 5;
        reconstructBindings[5].descriptorType = vk::DescriptorType::eSampler;
        reconstructBindings[5].descriptorCount = 1;
        reconstructBindings[5].stageFlags = vk::ShaderStageFlagBits::eFragment;
        reconstructBindings[6].binding = 6;
        reconstructBindings[6].descriptorType = vk::DescriptorType::eUniformBuffer;
        reconstructBindings[6].descriptorCount = 1;
        reconstructBindings[6].stageFlags = vk::ShaderStageFlagBits::eFragment;

        vk::DescriptorSetLayoutCreateInfo reconstructInfo{};
        reconstructInfo.bindingCount = static_cast<uint32_t>(reconstructBindings.size());
        reconstructInfo.pBindings = reconstructBindings.data();
        reconstructSetLayout = device.createDescriptorSetLayout(reconstructInfo);
    }

    vk::ShaderModule createShaderModule(const std::vector<uint32_t>& words) const
    {
        vk::ShaderModuleCreateInfo info{};
        info.codeSize = words.size() * sizeof(uint32_t);
        info.pCode = words.data();
        return device.createShaderModule(info);
    }

    void createPipelines()
    {
        const auto cubeSpv = readSpirv(std::filesystem::path(SHADER_DIR) / "cube.spv");
        const auto reconstructSpv = readSpirv(std::filesystem::path(SHADER_DIR) / "reconstruct.spv");
        const auto cubeModule = createShaderModule(cubeSpv);
        const auto reconstructModule = createShaderModule(reconstructSpv);

        vk::PipelineLayoutCreateInfo cubeLayoutInfo{};
        cubeLayoutInfo.setLayoutCount = 1;
        cubeLayoutInfo.pSetLayouts = &cubeSetLayout;
        cubePipelineLayout = device.createPipelineLayout(cubeLayoutInfo);

        vk::PipelineLayoutCreateInfo reconstructLayoutInfo{};
        reconstructLayoutInfo.setLayoutCount = 1;
        reconstructLayoutInfo.pSetLayouts = &reconstructSetLayout;
        reconstructPipelineLayout = device.createPipelineLayout(reconstructLayoutInfo);

        vk::PipelineShaderStageCreateInfo cubeStages[2];
        cubeStages[0].stage = vk::ShaderStageFlagBits::eVertex;
        cubeStages[0].module = cubeModule;
        cubeStages[0].pName = "vertMain";
        cubeStages[1].stage = vk::ShaderStageFlagBits::eFragment;
        cubeStages[1].module = cubeModule;
        cubeStages[1].pName = "fragMain";

        vk::VertexInputBindingDescription binding{};
        binding.stride = sizeof(Vertex);
        binding.inputRate = vk::VertexInputRate::eVertex;

        std::array<vk::VertexInputAttributeDescription, 3> attributes{};
        attributes[0].location = 0;
        attributes[0].format = vk::Format::eR32G32B32Sfloat;
        attributes[0].offset = offsetof(Vertex, position);
        attributes[1].location = 1;
        attributes[1].format = vk::Format::eR32G32B32Sfloat;
        attributes[1].offset = offsetof(Vertex, normal);
        attributes[2].location = 2;
        attributes[2].format = vk::Format::eR32G32B32Sfloat;
        attributes[2].offset = offsetof(Vertex, color);

        vk::PipelineVertexInputStateCreateInfo vertexInput{};
        vertexInput.vertexBindingDescriptionCount = 1;
        vertexInput.pVertexBindingDescriptions = &binding;
        vertexInput.vertexAttributeDescriptionCount = static_cast<uint32_t>(attributes.size());
        vertexInput.pVertexAttributeDescriptions = attributes.data();

        vk::PipelineInputAssemblyStateCreateInfo inputAssembly{};
        inputAssembly.topology = vk::PrimitiveTopology::eTriangleList;

        vk::PipelineViewportStateCreateInfo viewportState{};
        viewportState.viewportCount = 1;
        viewportState.scissorCount = 1;

        vk::PipelineRasterizationStateCreateInfo raster{};
        raster.polygonMode = vk::PolygonMode::eFill;
        raster.cullMode = vk::CullModeFlagBits::eBack;
        raster.frontFace = vk::FrontFace::eClockwise;
        raster.lineWidth = 1.0f;

        vk::PipelineMultisampleStateCreateInfo multisample{};
        multisample.rasterizationSamples = vk::SampleCountFlagBits::e1;

        vk::PipelineDepthStencilStateCreateInfo depthStencil{};
        depthStencil.depthTestEnable = VK_TRUE;
        depthStencil.depthWriteEnable = VK_TRUE;
        depthStencil.depthCompareOp = vk::CompareOp::eLess;

        vk::PipelineColorBlendAttachmentState blendAttachment{};
        blendAttachment.colorWriteMask = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                                         vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;

        vk::PipelineColorBlendStateCreateInfo colorBlend{};
        colorBlend.attachmentCount = 1;
        colorBlend.pAttachments = &blendAttachment;

        const vk::DynamicState dynamicStates[] = {vk::DynamicState::eViewport, vk::DynamicState::eScissor};
        vk::PipelineDynamicStateCreateInfo dynamic{};
        dynamic.dynamicStateCount = 2;
        dynamic.pDynamicStates = dynamicStates;

        vk::PipelineRenderingCreateInfo cubeRendering{};
        cubeRendering.colorAttachmentCount = 1;
        cubeRendering.pColorAttachmentFormats = &kOffscreenColorFormat;
        cubeRendering.depthAttachmentFormat = kDepthFormat;

        vk::GraphicsPipelineCreateInfo cubeInfo{};
        cubeInfo.pNext = &cubeRendering;
        cubeInfo.stageCount = 2;
        cubeInfo.pStages = cubeStages;
        cubeInfo.pVertexInputState = &vertexInput;
        cubeInfo.pInputAssemblyState = &inputAssembly;
        cubeInfo.pViewportState = &viewportState;
        cubeInfo.pRasterizationState = &raster;
        cubeInfo.pMultisampleState = &multisample;
        cubeInfo.pDepthStencilState = &depthStencil;
        cubeInfo.pColorBlendState = &colorBlend;
        cubeInfo.pDynamicState = &dynamic;
        cubeInfo.layout = cubePipelineLayout;
        cubePipeline = device.createGraphicsPipeline(nullptr, cubeInfo).value;

        vk::PipelineShaderStageCreateInfo reconstructStages[2];
        reconstructStages[0].stage = vk::ShaderStageFlagBits::eVertex;
        reconstructStages[0].module = reconstructModule;
        reconstructStages[0].pName = "vertMain";
        reconstructStages[1].stage = vk::ShaderStageFlagBits::eFragment;
        reconstructStages[1].module = reconstructModule;
        reconstructStages[1].pName = "fragMain";

        vk::PipelineVertexInputStateCreateInfo emptyVertexInput{};
        vk::PipelineDepthStencilStateCreateInfo noDepth{};
        vk::PipelineRenderingCreateInfo reconstructRendering{};
        reconstructRendering.colorAttachmentCount = 1;
        reconstructRendering.pColorAttachmentFormats = &swapchainFormat;

        vk::GraphicsPipelineCreateInfo reconstructInfo{};
        reconstructInfo.pNext = &reconstructRendering;
        reconstructInfo.stageCount = 2;
        reconstructInfo.pStages = reconstructStages;
        reconstructInfo.pVertexInputState = &emptyVertexInput;
        reconstructInfo.pInputAssemblyState = &inputAssembly;
        reconstructInfo.pViewportState = &viewportState;
        reconstructInfo.pRasterizationState = &raster;
        reconstructInfo.pMultisampleState = &multisample;
        reconstructInfo.pDepthStencilState = &noDepth;
        reconstructInfo.pColorBlendState = &colorBlend;
        reconstructInfo.pDynamicState = &dynamic;
        reconstructInfo.layout = reconstructPipelineLayout;
        reconstructPipeline = device.createGraphicsPipeline(nullptr, reconstructInfo).value;

        device.destroyShaderModule(cubeModule);
        device.destroyShaderModule(reconstructModule);
    }

    BufferResource createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties)
    {
        vk::BufferCreateInfo info{};
        info.size = size;
        info.usage = usage;
        info.sharingMode = vk::SharingMode::eExclusive;

        BufferResource resource;
        resource.buffer = device.createBuffer(info);
        const auto requirements = device.getBufferMemoryRequirements(resource.buffer);
        vk::MemoryAllocateInfo alloc{};
        alloc.allocationSize = requirements.size;
        alloc.memoryTypeIndex = findMemoryType(requirements.memoryTypeBits, properties);
        resource.memory = device.allocateMemory(alloc);
        device.bindBufferMemory(resource.buffer, resource.memory, 0);
        return resource;
    }

    void destroyBuffer(BufferResource& resource)
    {
        if (resource.buffer)
        {
            device.destroyBuffer(resource.buffer);
        }
        if (resource.memory)
        {
            device.freeMemory(resource.memory);
        }
        resource = {};
    }

    void createMeshBuffers()
    {
        std::vector<Vertex> vertices;
        std::vector<uint32_t> indices;
#ifdef SCENE_TOILET
        createToiletMesh(vertices, indices);
#else
        createCubeMesh(vertices, indices);
#endif
        indexCount = static_cast<uint32_t>(indices.size());

        const vk::DeviceSize vertexSize = vertices.size() * sizeof(Vertex);
        const vk::DeviceSize indexSize = indices.size() * sizeof(uint32_t);
        const auto hostFlags = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;

        vertexBuffer = createBuffer(vertexSize, vk::BufferUsageFlagBits::eVertexBuffer, hostFlags);
        indexBuffer = createBuffer(indexSize, vk::BufferUsageFlagBits::eIndexBuffer, hostFlags);

        void* vertexMapped = device.mapMemory(vertexBuffer.memory, 0, vertexSize);
        std::memcpy(vertexMapped, vertices.data(), static_cast<size_t>(vertexSize));
        device.unmapMemory(vertexBuffer.memory);

        void* indexMapped = device.mapMemory(indexBuffer.memory, 0, indexSize);
        std::memcpy(indexMapped, indices.data(), static_cast<size_t>(indexSize));
        device.unmapMemory(indexBuffer.memory);
    }

    void createUniformBuffers()
    {
        const auto hostFlags = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;
        cubeUboBuffer = createBuffer(sizeof(CubeUBO), vk::BufferUsageFlagBits::eUniformBuffer, hostFlags);
        reconstructUboBuffer = createBuffer(sizeof(ReconstructUBO), vk::BufferUsageFlagBits::eUniformBuffer, hostFlags);
        cubeUboMapped = device.mapMemory(cubeUboBuffer.memory, 0, sizeof(CubeUBO));
        reconstructUboMapped = device.mapMemory(reconstructUboBuffer.memory, 0, sizeof(ReconstructUBO));
    }

    void createDescriptorPoolAndSets()
    {
        const std::array<vk::DescriptorPoolSize, 3> poolSizes = {
            vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, 4},
            vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage, 8},
            vk::DescriptorPoolSize{vk::DescriptorType::eSampler, 4},
        };
        vk::DescriptorPoolCreateInfo poolInfo{};
        poolInfo.maxSets = 4;
        poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
        poolInfo.pPoolSizes = poolSizes.data();
        descriptorPool = device.createDescriptorPool(poolInfo);

        vk::DescriptorSetAllocateInfo cubeAlloc{};
        cubeAlloc.descriptorPool = descriptorPool;
        cubeAlloc.descriptorSetCount = 1;
        cubeAlloc.pSetLayouts = &cubeSetLayout;
        cubeDescriptorSet = device.allocateDescriptorSets(cubeAlloc).front();

        const vk::DescriptorSetLayout reconstructLayouts[2] = {reconstructSetLayout, reconstructSetLayout};
        vk::DescriptorSetAllocateInfo reconstructAlloc{};
        reconstructAlloc.descriptorPool = descriptorPool;
        reconstructAlloc.descriptorSetCount = 2;
        reconstructAlloc.pSetLayouts = reconstructLayouts;
        const auto sets = device.allocateDescriptorSets(reconstructAlloc);
        reconstructDescriptorSets[0] = sets[0];
        reconstructDescriptorSets[1] = sets[1];

        vk::DescriptorBufferInfo cubeBufferInfo{};
        cubeBufferInfo.buffer = cubeUboBuffer.buffer;
        cubeBufferInfo.range = sizeof(CubeUBO);
        vk::WriteDescriptorSet cubeWrite{};
        cubeWrite.dstSet = cubeDescriptorSet;
        cubeWrite.dstBinding = 0;
        cubeWrite.descriptorCount = 1;
        cubeWrite.descriptorType = vk::DescriptorType::eUniformBuffer;
        cubeWrite.pBufferInfo = &cubeBufferInfo;
        device.updateDescriptorSets(cubeWrite, {});

        for (uint32_t i = 0; i < 2; ++i)
        {
            const uint32_t current = i;
            const uint32_t previous = 1 - i;
            vk::DescriptorImageInfo images[4];
            images[0].imageView = offscreen[current].color.view;
            images[0].imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
            images[1].imageView = offscreen[current].depth.view;
            images[1].imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
            images[2].imageView = offscreen[previous].color.view;
            images[2].imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
            images[3].imageView = offscreen[previous].depth.view;
            images[3].imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

            vk::DescriptorImageInfo samplers[2];
            samplers[0].sampler = linearSampler;
            samplers[1].sampler = nearestSampler;

            vk::DescriptorBufferInfo reconstructBufferInfo{};
            reconstructBufferInfo.buffer = reconstructUboBuffer.buffer;
            reconstructBufferInfo.range = sizeof(ReconstructUBO);

            std::array<vk::WriteDescriptorSet, 7> writes{};
            for (uint32_t binding = 0; binding < 4; ++binding)
            {
                writes[binding].dstSet = reconstructDescriptorSets[i];
                writes[binding].dstBinding = binding;
                writes[binding].descriptorCount = 1;
                writes[binding].descriptorType = vk::DescriptorType::eSampledImage;
                writes[binding].pImageInfo = &images[binding];
            }
            writes[4].dstSet = reconstructDescriptorSets[i];
            writes[4].dstBinding = 4;
            writes[4].descriptorCount = 1;
            writes[4].descriptorType = vk::DescriptorType::eSampler;
            writes[4].pImageInfo = &samplers[0];
            writes[5].dstSet = reconstructDescriptorSets[i];
            writes[5].dstBinding = 5;
            writes[5].descriptorCount = 1;
            writes[5].descriptorType = vk::DescriptorType::eSampler;
            writes[5].pImageInfo = &samplers[1];
            writes[6].dstSet = reconstructDescriptorSets[i];
            writes[6].dstBinding = 6;
            writes[6].descriptorCount = 1;
            writes[6].descriptorType = vk::DescriptorType::eUniformBuffer;
            writes[6].pBufferInfo = &reconstructBufferInfo;
            device.updateDescriptorSets(writes, {});
        }
    }

    void createCommandObjects()
    {
        vk::CommandPoolCreateInfo poolInfo{};
        poolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
        poolInfo.queueFamilyIndex = graphicsFamily;
        commandPool = device.createCommandPool(poolInfo);

        vk::CommandBufferAllocateInfo alloc{};
        alloc.commandPool = commandPool;
        alloc.level = vk::CommandBufferLevel::ePrimary;
        alloc.commandBufferCount = 1;
        commandBuffer = device.allocateCommandBuffers(alloc).front();
    }

    void createSyncObjects()
    {
        imageAvailable = device.createSemaphore({});
        inFlight = device.createFence({vk::FenceCreateFlagBits::eSignaled});
        renderFinished.resize(swapchainImages.size());
        for (auto& semaphore : renderFinished)
        {
            semaphore = device.createSemaphore({});
        }
    }

    void recreateSwapchain()
    {
        int width = 0;
        int height = 0;
        glfwGetFramebufferSize(window, &width, &height);
        while (width == 0 || height == 0)
        {
            glfwGetFramebufferSize(window, &width, &height);
            glfwWaitEvents();
        }

        device.waitIdle();
        cleanupSwapchain();
        createSwapchain();
        renderFinished.resize(swapchainImages.size());
        for (auto& semaphore : renderFinished)
        {
            semaphore = device.createSemaphore({});
        }
        framebufferResized = false;
    }

    void cleanupSwapchain()
    {
        for (auto semaphore : renderFinished)
        {
            device.destroySemaphore(semaphore);
        }
        renderFinished.clear();
        for (auto view : swapchainViews)
        {
            device.destroyImageView(view);
        }
        swapchainViews.clear();
        if (swapchain)
        {
            device.destroySwapchainKHR(swapchain);
            swapchain = nullptr;
        }
    }

    void updateUniforms()
    {
        const float time = static_cast<float>(glfwGetTime());
        glm::mat4 model(1.0f);
#ifdef SCENE_TOILET
        model = glm::rotate(model, time * 0.55f, glm::vec3(0.0f, 1.0f, 0.0f));
        const glm::mat4 view = glm::lookAt(glm::vec3(1.95f, 1.45f, 2.55f), glm::vec3(0.0f, 0.05f, 0.0f),
                                          glm::vec3(0.0f, 1.0f, 0.0f));
#else
        model = glm::rotate(model, time * 0.65f, glm::vec3(0.0f, 1.0f, 0.0f));
        model = glm::rotate(model, time * 0.28f, glm::vec3(1.0f, 0.0f, 0.0f));
        const glm::mat4 view = glm::lookAt(glm::vec3(1.7f, 1.25f, 3.15f), glm::vec3(0.0f), glm::vec3(0.0f, 1.0f, 0.0f));
#endif
        glm::mat4 proj = glm::perspective(glm::radians(45.0f),
                                          static_cast<float>(kRenderWidth) / static_cast<float>(kRenderHeight),
                                          0.1f, 32.0f);
        proj[1][1] *= -1.0f;

        const float jitter = (frameIndex & 1u) ? 0.5f : -0.5f;
        glm::mat4 jitteredProj = proj;
        jitteredProj[3][0] += jitter * 2.0f / static_cast<float>(kRenderWidth);
        jitteredProj[3][1] += jitter * 2.0f / static_cast<float>(kRenderHeight);

        const glm::mat4 viewProj = jitteredProj * view;
        const glm::mat4 mvp = viewProj * model;

        CubeUBO cube{};
        cube.mvp = mvp;
        cube.model = model;
        cube.lightDir = glm::vec4(glm::normalize(glm::vec3(0.45f, 0.85f, 0.35f)), 0.0f);
        std::memcpy(cubeUboMapped, &cube, sizeof(cube));

        ReconstructUBO reconstruct{};
        reconstruct.invViewProj = glm::inverse(viewProj);
        reconstruct.prevViewProj = firstFrame ? viewProj : prevViewProj;
        reconstruct.objectReproject = firstFrame ? viewProj : (prevViewProj * prevModel * glm::inverse(model));
        reconstruct.renderSize = glm::vec2(kRenderWidth, kRenderHeight);
        reconstruct.outputSize = glm::vec2(swapchainExtent.width, swapchainExtent.height);
        reconstruct.frameIndex = frameIndex;
        reconstruct.firstFrame = firstFrame ? 1u : 0u;
        reconstruct.mode = checkerboardEnabled ? 1u : 0u;
        std::memcpy(reconstructUboMapped, &reconstruct, sizeof(reconstruct));

        prevViewProj = viewProj;
        prevModel = model;
    }

    void transitionImage(vk::CommandBuffer cmd, vk::Image image, vk::ImageLayout oldLayout, vk::ImageLayout newLayout,
                         vk::ImageAspectFlags aspect, vk::AccessFlags srcAccess, vk::AccessFlags dstAccess,
                         vk::PipelineStageFlags srcStage, vk::PipelineStageFlags dstStage)
    {
        vk::ImageMemoryBarrier barrier{};
        barrier.srcAccessMask = srcAccess;
        barrier.dstAccessMask = dstAccess;
        barrier.oldLayout = oldLayout;
        barrier.newLayout = newLayout;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.image = image;
        barrier.subresourceRange.aspectMask = aspect;
        barrier.subresourceRange.levelCount = 1;
        barrier.subresourceRange.layerCount = 1;
        cmd.pipelineBarrier(srcStage, dstStage, {}, {}, {}, barrier);
    }

    void recordCommandBuffer(uint32_t imageIndex)
    {
        commandBuffer.reset();
        commandBuffer.begin({vk::CommandBufferUsageFlagBits::eOneTimeSubmit});

        OffscreenTarget& current = offscreen[currentTarget];
        const vk::ImageLayout currentColorOld =
            firstFrame ? vk::ImageLayout::eUndefined : vk::ImageLayout::eShaderReadOnlyOptimal;
        const vk::ImageLayout currentDepthOld =
            firstFrame ? vk::ImageLayout::eUndefined : vk::ImageLayout::eShaderReadOnlyOptimal;

        transitionImage(commandBuffer, current.color.image, currentColorOld, vk::ImageLayout::eColorAttachmentOptimal,
                        vk::ImageAspectFlagBits::eColor, vk::AccessFlagBits::eShaderRead,
                        vk::AccessFlagBits::eColorAttachmentWrite, vk::PipelineStageFlagBits::eFragmentShader,
                        vk::PipelineStageFlagBits::eColorAttachmentOutput);
        transitionImage(commandBuffer, current.depth.image, currentDepthOld,
                        vk::ImageLayout::eDepthStencilAttachmentOptimal, vk::ImageAspectFlagBits::eDepth,
                        vk::AccessFlagBits::eShaderRead, vk::AccessFlagBits::eDepthStencilAttachmentWrite,
                        vk::PipelineStageFlagBits::eFragmentShader, vk::PipelineStageFlagBits::eEarlyFragmentTests);

        vk::ClearValue colorClear;
        colorClear.color = vk::ClearColorValue(std::array<float, 4>{0.045f, 0.05f, 0.08f, 1.0f});
        vk::ClearValue depthClear;
        depthClear.depthStencil = vk::ClearDepthStencilValue(1.0f, 0);

        vk::RenderingAttachmentInfo colorAttachment{};
        colorAttachment.imageView = current.color.view;
        colorAttachment.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
        colorAttachment.loadOp = vk::AttachmentLoadOp::eClear;
        colorAttachment.storeOp = vk::AttachmentStoreOp::eStore;
        colorAttachment.clearValue = colorClear;

        vk::RenderingAttachmentInfo depthAttachment{};
        depthAttachment.imageView = current.depth.view;
        depthAttachment.imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal;
        depthAttachment.loadOp = vk::AttachmentLoadOp::eClear;
        depthAttachment.storeOp = vk::AttachmentStoreOp::eStore;
        depthAttachment.clearValue = depthClear;

        vk::RenderingInfo cubePass{};
        cubePass.renderArea.extent = vk::Extent2D{kRenderWidth, kRenderHeight};
        cubePass.layerCount = 1;
        cubePass.colorAttachmentCount = 1;
        cubePass.pColorAttachments = &colorAttachment;
        cubePass.pDepthAttachment = &depthAttachment;
        commandBuffer.beginRendering(cubePass);

        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, cubePipeline);
        vk::Viewport cubeViewport{0.0f, 0.0f, static_cast<float>(kRenderWidth), static_cast<float>(kRenderHeight), 0.0f, 1.0f};
        vk::Rect2D cubeScissor{{0, 0}, {kRenderWidth, kRenderHeight}};
        commandBuffer.setViewport(0, cubeViewport);
        commandBuffer.setScissor(0, cubeScissor);
        commandBuffer.bindVertexBuffers(0, vertexBuffer.buffer, {0});
        commandBuffer.bindIndexBuffer(indexBuffer.buffer, 0, vk::IndexType::eUint32);
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, cubePipelineLayout, 0, cubeDescriptorSet, {});
        commandBuffer.drawIndexed(indexCount, 1, 0, 0, 0);
        commandBuffer.endRendering();

        transitionImage(commandBuffer, current.color.image, vk::ImageLayout::eColorAttachmentOptimal,
                        vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageAspectFlagBits::eColor,
                        vk::AccessFlagBits::eColorAttachmentWrite, vk::AccessFlagBits::eShaderRead,
                        vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eFragmentShader);
        transitionImage(commandBuffer, current.depth.image, vk::ImageLayout::eDepthStencilAttachmentOptimal,
                        vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageAspectFlagBits::eDepth,
                        vk::AccessFlagBits::eDepthStencilAttachmentWrite, vk::AccessFlagBits::eShaderRead,
                        vk::PipelineStageFlagBits::eLateFragmentTests, vk::PipelineStageFlagBits::eFragmentShader);

        if (firstFrame)
        {
            OffscreenTarget& previous = offscreen[1 - currentTarget];
            transitionImage(commandBuffer, previous.color.image, vk::ImageLayout::eUndefined,
                            vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageAspectFlagBits::eColor, {},
                            vk::AccessFlagBits::eShaderRead, vk::PipelineStageFlagBits::eTopOfPipe,
                            vk::PipelineStageFlagBits::eFragmentShader);
            transitionImage(commandBuffer, previous.depth.image, vk::ImageLayout::eUndefined,
                            vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageAspectFlagBits::eDepth, {},
                            vk::AccessFlagBits::eShaderRead, vk::PipelineStageFlagBits::eTopOfPipe,
                            vk::PipelineStageFlagBits::eFragmentShader);
        }

        transitionImage(commandBuffer, swapchainImages[imageIndex], vk::ImageLayout::eUndefined,
                        vk::ImageLayout::eColorAttachmentOptimal, vk::ImageAspectFlagBits::eColor, {},
                        vk::AccessFlagBits::eColorAttachmentWrite, vk::PipelineStageFlagBits::eTopOfPipe,
                        vk::PipelineStageFlagBits::eColorAttachmentOutput);

        vk::RenderingAttachmentInfo swapAttachment{};
        swapAttachment.imageView = swapchainViews[imageIndex];
        swapAttachment.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
        swapAttachment.loadOp = vk::AttachmentLoadOp::eClear;
        swapAttachment.storeOp = vk::AttachmentStoreOp::eStore;
        swapAttachment.clearValue = colorClear;

        vk::RenderingInfo resolvePass{};
        resolvePass.renderArea.extent = swapchainExtent;
        resolvePass.layerCount = 1;
        resolvePass.colorAttachmentCount = 1;
        resolvePass.pColorAttachments = &swapAttachment;
        commandBuffer.beginRendering(resolvePass);

        commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics, reconstructPipeline);
        vk::Viewport outputViewport{0.0f, 0.0f, static_cast<float>(swapchainExtent.width),
                                    static_cast<float>(swapchainExtent.height), 0.0f, 1.0f};
        vk::Rect2D outputScissor{{0, 0}, swapchainExtent};
        commandBuffer.setViewport(0, outputViewport);
        commandBuffer.setScissor(0, outputScissor);
        commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, reconstructPipelineLayout, 0,
                                         reconstructDescriptorSets[currentTarget], {});
        commandBuffer.draw(3, 1, 0, 0);
        commandBuffer.endRendering();

        transitionImage(commandBuffer, swapchainImages[imageIndex], vk::ImageLayout::eColorAttachmentOptimal,
                        vk::ImageLayout::ePresentSrcKHR, vk::ImageAspectFlagBits::eColor,
                        vk::AccessFlagBits::eColorAttachmentWrite, {},
                        vk::PipelineStageFlagBits::eColorAttachmentOutput, vk::PipelineStageFlagBits::eBottomOfPipe);
        commandBuffer.end();
    }

    void drawFrame()
    {
        (void)device.waitForFences(inFlight, VK_TRUE, std::numeric_limits<uint64_t>::max());

        uint32_t imageIndex = 0;
        vk::Result acquireResult = device.acquireNextImageKHR(swapchain, std::numeric_limits<uint64_t>::max(),
                                                              imageAvailable, nullptr, &imageIndex);
        if (acquireResult == vk::Result::eErrorOutOfDateKHR)
        {
            recreateSwapchain();
            return;
        }
        if (acquireResult != vk::Result::eSuccess && acquireResult != vk::Result::eSuboptimalKHR)
        {
            throw std::runtime_error("failed to acquire swapchain image");
        }

        device.resetFences(inFlight);
        updateUniforms();
        recordCommandBuffer(imageIndex);

        vk::PipelineStageFlags waitStage = vk::PipelineStageFlagBits::eColorAttachmentOutput;
        vk::SubmitInfo submit{};
        submit.waitSemaphoreCount = 1;
        submit.pWaitSemaphores = &imageAvailable;
        submit.pWaitDstStageMask = &waitStage;
        submit.commandBufferCount = 1;
        submit.pCommandBuffers = &commandBuffer;
        submit.signalSemaphoreCount = 1;
        submit.pSignalSemaphores = &renderFinished[imageIndex];
        graphicsQueue.submit(submit, inFlight);

        vk::PresentInfoKHR present{};
        present.waitSemaphoreCount = 1;
        present.pWaitSemaphores = &renderFinished[imageIndex];
        present.swapchainCount = 1;
        present.pSwapchains = &swapchain;
        present.pImageIndices = &imageIndex;
        const vk::Result presentResult = presentQueue.presentKHR(present);
        if (presentResult == vk::Result::eErrorOutOfDateKHR || presentResult == vk::Result::eSuboptimalKHR ||
            framebufferResized)
        {
            recreateSwapchain();
        }
        else if (presentResult != vk::Result::eSuccess)
        {
            throw std::runtime_error("failed to present swapchain image");
        }

        currentTarget = 1 - currentTarget;
        firstFrame = false;
        ++frameIndex;
    }

    void pollToggle()
    {
        const bool spaceDown = glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS;
        if (spaceDown && !spaceWasDown)
        {
            checkerboardEnabled = !checkerboardEnabled;
            glfwSetWindowTitle(window, checkerboardEnabled ? kWindowTitleCheckerboard : kWindowTitleBilinear);
        }
        spaceWasDown = spaceDown;
    }

    void mainLoop()
    {
        while (!glfwWindowShouldClose(window))
        {
            glfwPollEvents();
            if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            {
                glfwSetWindowShouldClose(window, GLFW_TRUE);
            }
            pollToggle();
            drawFrame();
        }
        device.waitIdle();
    }

    void cleanup()
    {
        if (device)
        {
            device.waitIdle();
            device.destroyFence(inFlight);
            device.destroySemaphore(imageAvailable);
            cleanupSwapchain();
            device.destroyCommandPool(commandPool);
            device.destroyDescriptorPool(descriptorPool);
            if (cubeUboMapped)
            {
                device.unmapMemory(cubeUboBuffer.memory);
            }
            if (reconstructUboMapped)
            {
                device.unmapMemory(reconstructUboBuffer.memory);
            }
            destroyBuffer(cubeUboBuffer);
            destroyBuffer(reconstructUboBuffer);
            destroyBuffer(vertexBuffer);
            destroyBuffer(indexBuffer);
            device.destroySampler(linearSampler);
            device.destroySampler(nearestSampler);
            for (auto& target : offscreen)
            {
                destroyImage(target.color);
                destroyImage(target.depth);
            }
            device.destroyPipeline(cubePipeline);
            device.destroyPipeline(reconstructPipeline);
            device.destroyPipelineLayout(cubePipelineLayout);
            device.destroyPipelineLayout(reconstructPipelineLayout);
            device.destroyDescriptorSetLayout(cubeSetLayout);
            device.destroyDescriptorSetLayout(reconstructSetLayout);
            device.destroy();
        }
        if (instance)
        {
            instance.destroySurfaceKHR(surface);
            if (debugMessenger && pfnDestroyDebugMessenger)
            {
                pfnDestroyDebugMessenger(instance, debugMessenger, nullptr);
            }
            instance.destroy();
        }
        if (window)
        {
            glfwDestroyWindow(window);
        }
        glfwTerminate();
    }
};

int main()
{
    try
    {
        CheckerboardApp app;
        app.run();
    }
    catch (const vk::SystemError& error)
    {
        std::cerr << "Vulkan error: " << error.what() << '\n';
        return 1;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
