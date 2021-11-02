#pragma once
#include <string>
#include <sstream>
#include <ctime>
#include <functional>
#include<stdio.h>


#ifndef COMMON_H
#define COMMON_H

#endif // COMMON_H

namespace patch
{
template < typename T > std::string to_string( const T& n )
{
    std::ostringstream stm ;
    stm << n ;
    return stm.str() ;
}
}

enum fidelityType {L2, linear, KL, SPG};

typedef std::pair<std::string, float> NameScale_t;

class GenericParameter
{
    public:
    std::string in_name, out_name, base_name, extension;
    int natureOfData;
    fidelityType fidelity;
    //std::vector< NameScale_t > v_coord_name_scale, v_attrib_name_scale;
    GenericParameter(std::string inName = "in_name", double reg_strength = 0, double fidelity = 0)
    {
        this->in_name  = inName;
        char buffer [inName.size() + 10];
        std::string extension = inName.substr(inName.find_last_of(".") + 1);
        this->extension  = extension;
        std::string baseName  = inName.subs