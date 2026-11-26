program reference_functional_neighbor_graphs_driver
 use sklearn_common,only:dp,require
 use sklearn_neighbors,only:csr_matrix,nearest_neighbors,kneighbors_classifier,kneighbors_regressor, &
  radius_neighbors_classifier,radius_neighbors_regressor,kneighbors_graph,radius_neighbors_graph,sort_graph_by_row_values
 implicit none
 type(csr_matrix)::graph,s
 type(nearest_neighbors)::nn,nn0
 type(kneighbors_classifier)::kc,kc0
 type(kneighbors_regressor)::kr,kr0
 type(radius_neighbors_classifier)::rc,rc0
 type(radius_neighbors_regressor)::rr,rr0
 real(dp),allocatable::x(:,:)
 real(dp)::radius,power,fit_power
 integer::operation,n,p,storage,family,k,include_mode,fit_kind,mutation,i,ios,case_id,copy_flag,warn_flag
 character(16)::mode,metric,fit_metric
 logical::sorted
 case_id=0
 do
  read(*,*,iostat=ios)operation
  if(ios/=0)exit
  if(operation==0)then
   read(*,*)n,p,storage,family,k,radius,include_mode,fit_kind,mutation
   read(*,*)mode,metric,power,fit_metric,fit_power
   call require(include_mode>=0.and.include_mode<=2,'invalid include_self control')
   if(storage==0.or.storage==2)then
    allocate(x(n,p))
    do i=1,n
     read(*,*)x(i,:)
    end do
   else
    call read_csr(s,n,p)
   end if
   nn=nn0;kc=kc0;kr=kr0;rc=rc0;rr=rr0
   nn%metric=fit_metric;nn%p=fit_power
   kc%metric=fit_metric;kc%p=fit_power;kr%metric=fit_metric;kr%p=fit_power
   rc%metric=fit_metric;rc%p=fit_power;rr%metric=fit_metric;rr%p=fit_power
   if(fit_kind>0)then
    select case(fit_kind)
    case(1);call nn%fit(x)
    case(2)
     if(family==0)then
      call kc%fit(x,[(mod(i,2),i=1,n)])
     else
      call rc%fit(x,[(mod(i,2),i=1,n)])
     end if
    case(3)
     if(family==0)then
      call kr%fit(x,[(real(i,dp),i=1,n)])
     else
      call rr%fit(x,[(real(i,dp),i=1,n)])
     end if
    end select
   end if
   if(mutation==1)then
    nn%metric=metric;nn%p=power;kc%metric=metric;kc%p=power;kr%metric=metric;kr%p=power
    rc%metric=metric;rc%p=power;rr%metric=metric;rr%p=power
   end if
   if(family==0)then
    select case(abs(fit_kind))
    case(0)
     if(storage==0.or.storage==2)then
      if(include_mode==2)then
       graph=kneighbors_graph(x,k,mode,metric,power,'auto')
      else
       graph=kneighbors_graph(x,k,mode,metric,power,include_mode==1)
      end if
     else
      if(include_mode==2)then
       graph=kneighbors_graph(s,k,mode,metric,power,'auto')
      else
       graph=kneighbors_graph(s,k,mode,metric,power,include_mode==1)
      end if
     end if
    case(1)
      if(include_mode==2)then
       graph=kneighbors_graph(nn,k,mode,metric,power,'auto')
      else
       graph=kneighbors_graph(nn,k,mode,metric,power,include_mode==1)
      end if
    case(2)
      if(include_mode==2)then
       graph=kneighbors_graph(kc,k,mode,metric,power,'auto')
      else
       graph=kneighbors_graph(kc,k,mode,metric,power,include_mode==1)
      end if
    case(3)
      if(include_mode==2)then
       graph=kneighbors_graph(kr,k,mode,metric,power,'auto')
      else
       graph=kneighbors_graph(kr,k,mode,metric,power,include_mode==1)
      end if
    end select
   else if(family==1)then
    select case(abs(fit_kind))
    case(0)
     if(storage==0.or.storage==2)then
      if(include_mode==2)then
       graph=radius_neighbors_graph(x,radius,mode,metric,power,'auto')
      else
       graph=radius_neighbors_graph(x,radius,mode,metric,power,include_mode==1)
      end if
     else
      if(include_mode==2)then
       graph=radius_neighbors_graph(s,radius,mode,metric,power,'auto')
      else
       graph=radius_neighbors_graph(s,radius,mode,metric,power,include_mode==1)
      end if
     end if
    case(1)
      if(include_mode==2)then
       graph=radius_neighbors_graph(nn,radius,mode,metric,power,'auto')
      else
       graph=radius_neighbors_graph(nn,radius,mode,metric,power,include_mode==1)
      end if
    case(2)
      if(include_mode==2)then
       graph=radius_neighbors_graph(rc,radius,mode,metric,power,'auto')
      else
       graph=radius_neighbors_graph(rc,radius,mode,metric,power,include_mode==1)
      end if
    case(3)
      if(include_mode==2)then
       graph=radius_neighbors_graph(rr,radius,mode,metric,power,'auto')
      else
       graph=radius_neighbors_graph(rr,radius,mode,metric,power,include_mode==1)
      end if
    end select
   end if
   sorted=.false.
  else
   read(*,*)n,p,copy_flag,warn_flag
   call read_csr(s,n,p)
   graph=sort_graph_by_row_values(s,copy=copy_flag==1,warn_when_not_sorted=warn_flag==1,was_sorted=sorted)
  end if
  case_id=case_id+1;write(*,'(a,i0)')'CASE ',case_id
  write(*,'(*(i0,1x))')graph%n_rows,graph%n_cols,graph%nnz()
  write(*,'(*(i0,1x))')graph%indptr
  write(*,'(*(i0,1x))')graph%indices
  write(*,'(*(es26.17e3,1x))')graph%data
  write(*,'(*(es26.17e3,1x))')graph%to_dense()
  if(operation==1)then
   write(*,'(*(i0,1x))')s%indptr
   write(*,'(*(i0,1x))')s%indices
   write(*,'(*(es26.17e3,1x))')s%data
  else
   write(*,*);write(*,*);write(*,*)
  end if
  write(*,'(i0)')merge(1,0,sorted)
  if(allocated(x))deallocate(x)
 end do
contains
 subroutine read_csr(a,rows,cols)
  type(csr_matrix),intent(out)::a
  integer,intent(in)::rows,cols
  integer::nnz
  read(*,*)nnz;a%n_rows=rows;a%n_cols=cols
  allocate(a%indptr(rows+1),a%indices(nnz),a%data(nnz))
  read(*,*)a%indptr;read(*,*)a%indices;read(*,*)a%data
 end subroutine
end program
