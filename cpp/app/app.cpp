// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "ccf/app_interface.h"
#include "ccf/common_auth_policies.h"
#include "ccf/http_query.h"
#include "ccf/json_handler.h"

#include <nlohmann/json.hpp>
#define FMT_HEADER_ONLY
#include "ccf/ds/logger.h"
#include "crypto/entropy.h"
#include "crypto/key_pair.h"
#include "crypto/rsa_key_pair.h"

#include <cstring>
#include <fmt/format.h>
#include <iostream>
#include <openssl/bio.h>
#include <openssl/evp.h>
#include <vector>

namespace app
{
  std::vector<unsigned char> base64_decode(const std::string& input)
  {
    BIO *bio, *b64;
    std::vector<unsigned char> buffer(input.size());

    bio = BIO_new_mem_buf(input.c_str(), -1);
    b64 = BIO_new(BIO_f_base64());
    bio = BIO_push(b64, bio);

    // Decode the base64 string
    BIO_read(bio, buffer.data(), buffer.size());

    BIO_free_all(bio);

    return buffer;
  }
  double process_weights(const std::vector<unsigned char>& binary_weights)
  {
    // Example: Calculate the sum of weights
    double weight_sum = 0.0;

    // Iterate over the binary_weights vector and accumulate the sum
    for (const auto& weight : binary_weights)
    {
      weight_sum += static_cast<double>(weight);
    }

    return weight_sum;
  }

  // Key-value store types
  using Map = kv::Map<size_t, std::string>;
  static constexpr auto RECORDS = "records";
  using User = kv::Map<size_t, std::string>; // User information
  using Model = kv::Map<size_t, nlohmann::json>;
  using Weights = kv::Map<size_t, std::string>; // Model weights
  static constexpr auto USERS = "users";
  static constexpr auto MODELS = "models";
  static constexpr auto WEIGHTS = "weights";

  // API types
  struct Write
  {
    struct In
    {
      std::string msg; // Change the type to std::string
    };

    using Out = void;
  };

  struct GlobalModel
  {
    std::string model_name;
    nlohmann::json model_data;
  };

  struct ModelWrite
  {
    struct In
    {
      GlobalModel global_model; // Use the GlobalModel structure in ModelWrite
    };

    using Out = void;
  };

  struct ModelWeightWrite
  {
    struct In
    {
      size_t model_id;
      std::string weights_base64;
      size_t round_no;
    };

    using Out = void;
  };

  DECLARE_JSON_TYPE(ModelWeightWrite::In);
  DECLARE_JSON_REQUIRED_FIELDS(
    ModelWeightWrite::In, model_id, weights_base64, round_no);
  DECLARE_JSON_TYPE(Write::In);
  DECLARE_JSON_REQUIRED_FIELDS(Write::In, msg);

  // Declare JSON types for GlobalModel
  DECLARE_JSON_TYPE(GlobalModel);
  DECLARE_JSON_REQUIRED_FIELDS(GlobalModel, model_name, model_data);

  // Declare JSON types for ModelWrite
  DECLARE_JSON_TYPE(ModelWrite::In);
  DECLARE_JSON_REQUIRED_FIELDS(ModelWrite::In, global_model);

  class AppHandlers : public ccf::UserEndpointRegistry
  {
  public:
    AppHandlers(ccfapp::AbstractNodeContext& context) :
      ccf::UserEndpointRegistry(context)
    {
      openapi_info.title = "CCF Sample C++ App";
      openapi_info.description =
        "This minimal CCF C++ application aims to be "
        "used as a template for CCF developers.";
      openapi_info.document_version = "0.0.1";

      auto write = [this](auto& ctx, nlohmann::json&& params) {
        const auto parsed_query =
          http::parse_query(ctx.rpc_ctx->get_request_query());

        std::string error_reason;
        size_t id = 0;
        if (!http::get_query_value(parsed_query, "id", id, error_reason))
        {
          return ccf::make_error(
            HTTP_STATUS_BAD_REQUEST,
            ccf::errors::InvalidQueryParameterValue,
            std::move(error_reason));
        }

        const auto in = params.get<Write::In>();
        if (in.msg.empty())
        {
          return ccf::make_error(
            HTTP_STATUS_BAD_REQUEST,
            ccf::errors::InvalidInput,
            "Cannot record an empty log message.");
        }

        auto records_handle = ctx.tx.template rw<Map>(RECORDS);
        records_handle->put(id, in.msg);
        return ccf::make_success();
      };

      auto default_route = [this](ccf::endpoints::EndpointContext& ctx) {
        // Customize the behavior of the default route here
        return ccf::make_success("Welcome to the CCF Sample C++ App!");
      };

      auto read = [this](auto& ctx, nlohmann::json&& params) {
        const auto parsed_query =
          http::parse_query(ctx.rpc_ctx->get_request_query());

        std::string error_reason;
        size_t id = 0;
        if (!http::get_query_value(parsed_query, "id", id, error_reason))
        {
          return ccf::make_error(
            HTTP_STATUS_BAD_REQUEST,
            ccf::errors::InvalidQueryParameterValue,
            std::move(error_reason));
        }

        auto records_handle = ctx.tx.template ro<Map>(RECORDS);
        auto msg = records_handle->get(id);
        if (!msg.has_value())
        {
          return ccf::make_error(
            HTTP_STATUS_NOT_FOUND,
            ccf::errors::ResourceNotFound,
            fmt::format("Cannot find record for id \"{}\".", id));
        }
        return ccf::make_success(msg.value());
      };

      auto write_user =
        [this](ccf::endpoints::EndpointContext& ctx, nlohmann::json&& params) {
          const auto in = params.get<Write::In>();
          if (in.msg.empty())
          {
            return ccf::make_error(
              HTTP_STATUS_BAD_REQUEST,
              ccf::errors::InvalidInput,
              "Cannot record an empty user message.");
          }

          auto users_handle = ctx.tx.template rw<User>(USERS);
          users_handle->put(users_handle->size(), in.msg);
          return ccf::make_success();
        };

      // auto write_model =
      //   [this](ccf::endpoints::EndpointContext& ctx, nlohmann::json&& params)
      //   {
      //     const auto in = params.get<Write::In>();
      //     if (in.msg.empty())
      //     {
      //       return ccf::make_error(
      //         HTTP_STATUS_BAD_REQUEST,
      //         ccf::errors::InvalidInput,
      //         "Cannot record an empty model message.");
      //     }

      //     auto models_handle = ctx.tx.template rw<Model>(MODELS);
      //     size_t model_id = models_handle->size();
      //     models_handle->put(model_id, nlohmann::json::parse(in.msg));

      //     CCF_APP_INFO("x is currently {}", in.msg);
      //     nlohmann::json payload = {
      //       {"model_id", model_id}, {"message", "Model uploaded
      //       successfully"}};
      //     auto response = ccf::make_success(std::move(payload));
      //     return response;
      //   };
      auto write_model =
        [this](ccf::endpoints::EndpointContext& ctx, nlohmann::json&& params) {
          try
          {
            CCF_APP_INFO("uploading endpoint called");
            const auto in =
              params.get<ModelWrite::In>(); // Change the parameter type
            const GlobalModel& global_model = in.global_model;

            if (
              global_model.model_name.empty() ||
              global_model.model_data.is_null())
            {
              return ccf::make_error(
                HTTP_STATUS_BAD_REQUEST,
                ccf::errors::InvalidInput,
                "Invalid or empty model payload.");
            }

            auto models_handle = ctx.tx.template rw<Model>(MODELS);
            size_t model_id = models_handle->size();

            // Save the model data in the Model map
            models_handle->put(model_id, global_model.model_data);

            // Example: Extract and print information from the custom model
            // structure
            std::string model_name = global_model.model_name;
            CCF_APP_INFO("Model ID: {}, Model Name: {}", model_id, model_name);

            nlohmann::json payload = {
              {"model_id", model_id},
              {"message", "Model uploaded successfully"},
              {"model_name", model_name}};

            auto response = ccf::make_success(std::move(payload));
            return response;
          }
          catch (const std::exception& e)
          {
            CCF_APP_INFO("Exception in write_model: {}", e.what());

            // Handle the exception and return an error response
            return ccf::make_error(
              HTTP_STATUS_INTERNAL_SERVER_ERROR,
              ccf::errors::InternalError,
              "Internal server error occurred while processing the request.");
          }
          catch (...)
          {
            CCF_APP_INFO("Unknown exception in write_model");

            // Handle the unknown exception and return an error response
            return ccf::make_error(
              HTTP_STATUS_INTERNAL_SERVER_ERROR,
              ccf::errors::InternalError,
              "Internal server error occurred while processing the request.");
          }
        };

      auto write_weights =
        [this](ccf::endpoints::EndpointContext& ctx, nlohmann::json&& params) {
          try
          {
            CCF_APP_INFO("model weight write endpoint called");
            const auto in = params.get<ModelWeightWrite::In>();

            // if (
            //   in.model_id==0 || in.weights_base64.empty() ||
            //   in.round_no == 0)
            // {
            //   return ccf::make_error(
            //     HTTP_STATUS_BAD_REQUEST,
            //     ccf::errors::InvalidInput,
            //     "Invalid or empty model weight payload.");
            // }

            auto weights_handle = ctx.tx.template rw<Weights>(WEIGHTS);
            weights_handle->put(weights_handle->size(), in.weights_base64);

            // Handle the model weight information as needed
            // CCF_APP_INFO(
            //   "Model ID: {}, Round No: {}, Weights: {}",
            //   in.model_id,
            //   in.round_no,
            //   in.weights_base64);

            nlohmann::json payload = {{"message", "Model weights received"}};
            auto response = ccf::make_success(std::move(payload));
            return response;
          }
          catch (const std::exception& e)
          {
            CCF_APP_INFO("Exception in write_weights: {}", e.what());

            // Handle the exception and return an error response
            return ccf::make_error(
              HTTP_STATUS_INTERNAL_SERVER_ERROR,
              ccf::errors::InternalError,
              "Internal server error occurred while processing the request.");
          }
          catch (...)
          {
            CCF_APP_INFO("Unknown exception in write_weights");

            // Handle the unknown exception and return an error response
            return ccf::make_error(
              HTTP_STATUS_INTERNAL_SERVER_ERROR,
              ccf::errors::InternalError,
              "Internal server error occurred while processing the request.");
          }
        };

      // auto aggregate_weights = [this](auto& ctx, nlohmann::json&& params) {
      //   auto start_time = std::chrono::steady_clock::now();

      //   auto models_handle = ctx.tx.template ro<Model>(MODELS);
      //   auto weights_handle = ctx.tx.template ro<Weights>(WEIGHTS);

      //   // Perform aggregations on weights
      //   // Example: Summing up all weights
      //   double total_weight = 0.0;

      //   models_handle->foreach(
      //     [&](
      //       const size_t& model_id, const nlohmann::json& model_json) -> bool
      //       {
      //       // Your code to process each model entry
      //       CCF_APP_TRACE(
      //         "Model ID: {}, Model JSON: {}", model_id, model_json.dump());

      //       // Extract the 'weights' field from the model_json and accumulate
      //       // the values
      //       if (model_json.contains("weights"))
      //       {
      //         const auto& weights = model_json["weights"];
      //         if (weights.is_object())
      //         {
      //           for (auto it = weights.begin(); it != weights.end(); ++it)
      //           {
      //             if (it.value().is_number())
      //             {
      //               total_weight +=
      //                 it.value().get<double>(); // Explicit cast to double
      //             }
      //           }
      //         }
      //       }

      //       // Return true to continue iteration, false to stop
      //       return true;
      //     });

      //   auto end_time = std::chrono::steady_clock::now();
      //   auto elapsed_time =
      //     std::chrono::duration_cast<std::chrono::milliseconds>(
      //       end_time - start_time)
      //       .count();

      //   CCF_APP_INFO("Aggregation Time: {} ms", elapsed_time);

      //   return ccf::make_success(total_weight);
      // };
     auto aggregate_weights = [this](auto& ctx, nlohmann::json&& params) {
    auto start_time = std::chrono::steady_clock::now();

    const auto parsed_query =
        http::parse_query(ctx.rpc_ctx->get_request_query());

    std::string error_reason;
    size_t model_id = 0;
    if (!http::get_query_value(
            parsed_query, "model_id", model_id, error_reason))
    {
        return ccf::make_error(
            HTTP_STATUS_BAD_REQUEST,
            ccf::errors::InvalidQueryParameterValue,
            std::move(error_reason));
    }

    auto weights_handle = ctx.tx.template ro<Weights>(WEIGHTS);

    // Perform aggregations on weights
    double total_weight_sum = 0.0;
    size_t total_weights_count = 0;

    weights_handle->foreach(
        [&](const size_t& weight_id, const std::string& base64_weights) -> bool {
            // Check if the weight record is associated with the specified model_id
            // (You may need to adjust your data model accordingly)
            // For example, you might store the model_id in the weight record.
            // Here, I assume the weight_id itself represents the model_id.
            if (weight_id == model_id) {
                // Decode base64-encoded weights
                std::vector<unsigned char> binary_weights = base64_decode(base64_weights);

                // Process the weights (you may need to implement a proper
                // processing logic)
                double weight_sum = process_weights(binary_weights);

                // Accumulate the weights
                total_weight_sum += weight_sum;
                total_weights_count++;
            }

            return true;
        });

    // Calculate the average weight
    double average_weight = (total_weights_count > 0) ?
        total_weight_sum / total_weights_count :
        0.0;

    auto end_time = std::chrono::steady_clock::now();
    auto elapsed_time =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time)
            .count();

    CCF_APP_INFO("Aggregation Time: {} ms", elapsed_time);

    return ccf::make_success(average_weight);
};

      auto get_model = [this](auto& ctx, nlohmann::json&& params) {
        const auto parsed_query =
          http::parse_query(ctx.rpc_ctx->get_request_query());
        std::string error_reason;
        size_t model_id = 0;
        if (!http::get_query_value(
              parsed_query, "model_id", model_id, error_reason))
        {
          return ccf::make_error(
            HTTP_STATUS_BAD_REQUEST,
            ccf::errors::InvalidQueryParameterValue,
            std::move(error_reason));
        }
        CCF_APP_INFO("params id {}", model_id);
        auto models_handle = ctx.tx.template ro<Model>(MODELS);
        auto model_json = models_handle->get(model_id);
        if (!model_json.has_value())
        {
          return ccf::make_error(
            HTTP_STATUS_NOT_FOUND,
            ccf::errors::ResourceNotFound,
            fmt::format("Cannot find model for id \"{}\".", model_id));
        }

        nlohmann::json payload = {
          {"model_id", model_id},
          {"message", "Model details retrieved successfully"},
          {"model_details", std::move(model_json)}};
        auto response = ccf::make_success(std::move(payload));
        return response;
      };

      make_endpoint(
        "/user",
        HTTP_POST,
        ccf::json_adapter(write_user),
        ccf::no_auth_required)
        .set_auto_schema<Write::In, void>()
        .install();

      make_endpoint(
        "/model",
        HTTP_POST,
        ccf::json_adapter(write_model),
        ccf::no_auth_required)
        .set_auto_schema<ModelWrite::In, void>() // Set auto schema for
        .install();
      make_endpoint(
        "/weights",
        HTTP_POST,
        ccf::json_adapter(write_weights),
        ccf::no_auth_required)
        .set_auto_schema<ModelWeightWrite::In, void>()
        .install();
      make_read_only_endpoint(
        "/aggregate_weights",
        HTTP_GET,
        ccf::json_read_only_adapter(aggregate_weights),
        ccf::no_auth_required)
        //  {ccf::user_cert_auth_policy})
        .add_query_parameter<size_t>("model_id")
        .install();
      make_read_only_endpoint(
        "/model",
        HTTP_GET,
        ccf::json_read_only_adapter(get_model),
        ccf::no_auth_required)
        .set_auto_schema<void, nlohmann::json>()
        .add_query_parameter<size_t>("model_id")
        .install();

      make_read_only_endpoint(
        "/log",
        HTTP_GET,
        ccf::json_read_only_adapter(read),
        ccf::no_auth_required)
        .set_auto_schema<void, void>()
        .add_query_parameter<size_t>("id")
        .install();
      make_endpoint(
        "/log", HTTP_POST, ccf::json_adapter(write), ccf::no_auth_required)
        .set_auto_schema<Write::In, void>()
        .add_query_parameter<size_t>("id")
        .install();
    }
  };
} // namespace app

namespace ccfapp
{
  std::unique_ptr<ccf::endpoints::EndpointRegistry> make_user_endpoints(
    ccfapp::AbstractNodeContext& context)
  {
    return std::make_unique<app::AppHandlers>(context);
  }
}