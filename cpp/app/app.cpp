// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include "ccf/app_interface.h"
#include "ccf/common_auth_policies.h"
#include "ccf/http_query.h"
#include "ccf/json_handler.h"

#include <nlohmann/json.hpp>
#define FMT_HEADER_ONLY
#include "ccf/ds/logger.h"

#include <fmt/format.h>
namespace app
{
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
  DECLARE_JSON_TYPE(Write::In);
  DECLARE_JSON_REQUIRED_FIELDS(Write::In, msg);

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

      auto write_model =
        [this](ccf::endpoints::EndpointContext& ctx, nlohmann::json&& params) {
          const auto in = params.get<Write::In>();
          if (in.msg.empty())
          {
            return ccf::make_error(
              HTTP_STATUS_BAD_REQUEST,
              ccf::errors::InvalidInput,
              "Cannot record an empty model message.");
          }

          auto models_handle = ctx.tx.template rw<Model>(MODELS);
          size_t model_id = models_handle->size();
          models_handle->put(model_id, nlohmann::json::parse(in.msg));

          CCF_APP_INFO("x is currently {}", in.msg);
          nlohmann::json payload = {
            {"model_id", model_id}, {"message", "Model uploaded successfully"}};
          auto response = ccf::make_success(std::move(payload));
          return response;
        };

      auto write_weights =
        [this](ccf::endpoints::EndpointContext& ctx, nlohmann::json&& params) {
          const auto in = params.get<Write::In>();
          if (in.msg.empty())
          {
            return ccf::make_error(
              HTTP_STATUS_BAD_REQUEST,
              ccf::errors::InvalidInput,
              "Cannot record empty weights.");
          }

          auto weights_handle = ctx.tx.template rw<Weights>(WEIGHTS);
          weights_handle->put(weights_handle->size(), in.msg);
          nlohmann::json payload = {{"message", "Weights received"}};
          auto response = ccf::make_success(std::move(payload));
          return response;
        };

      // auto aggregate_weights = [this](auto& ctx, nlohmann::json&& params) {
      //   auto models_handle = ctx.tx.template ro<Model>(MODELS);

      //   // Perform aggregations on weights
      //   // Example: Summing up all weights
      //   double total_weight = 0.0;

      //   models_handle->foreach(
      //     [&](
      //       const size_t& model_id, const nlohmann::json& model_json) -> bool
      //       {
      //       // Your code to process each model entry
      //       CCF_APP_INFO(
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

      //   return ccf::make_success(total_weight);
      // };
      auto aggregate_weights = [this](auto& ctx, nlohmann::json&& params) {
        auto start_time = std::chrono::steady_clock::now();

        auto models_handle = ctx.tx.template ro<Model>(MODELS);

        // Perform aggregations on weights
        // Example: Summing up all weights
        double total_weight = 0.0;

        models_handle->foreach(
          [&](
            const size_t& model_id, const nlohmann::json& model_json) -> bool {
            // Your code to process each model entry
            CCF_APP_INFO(
              "Model ID: {}, Model JSON: {}", model_id, model_json.dump());

            // Extract the 'weights' field from the model_json and accumulate
            // the values
            if (model_json.contains("weights"))
            {
              const auto& weights = model_json["weights"];
              if (weights.is_object())
              {
                for (auto it = weights.begin(); it != weights.end(); ++it)
                {
                  if (it.value().is_number())
                  {
                    total_weight +=
                      it.value().get<double>(); // Explicit cast to double
                  }
                }
              }
            }

            // Return true to continue iteration, false to stop
            return true;
          });

        auto end_time = std::chrono::steady_clock::now();
        auto elapsed_time =
          std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time)
            .count();

        CCF_APP_INFO("Aggregation Time: {} ms", elapsed_time);

        return ccf::make_success(total_weight);
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
        .set_auto_schema<Write::In, void>()
        .install();

      make_endpoint(
        "/weights",
        HTTP_POST,
        ccf::json_adapter(write_weights),
        ccf::no_auth_required)
        .set_auto_schema<Write::In, void>()
        .install();

      make_read_only_endpoint(
        "/aggregate_weights",
        HTTP_GET,
        ccf::json_read_only_adapter(aggregate_weights),
        ccf::no_auth_required)
        //  {ccf::user_cert_auth_policy})
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
} // namespace ccfapp